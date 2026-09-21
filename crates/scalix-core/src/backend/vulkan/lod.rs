//! Hierarchical LoD / Mipchain Downscaler Pipeline
//!
//! Generates multi-pass mip pyramid chains (by factors of 2x2) using hardware GPU blitters
//! to eliminate aliasing, high-frequency scintillation, and Moire artifacts during extreme
//! downscaling (e.g., 4K -> 360p / 320x320).

use std::sync::Arc;
use ash::vk;
use crate::backend::vulkan::blit::{to_vk_filter, to_vk_format_info};
use crate::backend::vulkan::context::VulkanContext;
use crate::backend::vulkan::util::{CommandBufferGuard, GpuBuffer, GpuImage, QueryPoolGuard};
use crate::backend::vulkan::{VulkanPipeline, VulkanStrategy};
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{ImageDesc, ImageDescMut, ResizeOptions, Result, ScalixError};

pub struct VulkanLodDownscaler {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
}

impl VulkanLodDownscaler {
    #[inline]
    #[must_use]
    pub fn new(ctx: Arc<VulkanContext>, profiler: Arc<dyn Profiler>) -> Self {
        Self { ctx, profiler }
    }

    pub fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, options: &ResizeOptions) -> Result<()> {
        let (vk_format, src_is_rgb) = to_vk_format_info(src.format)?;
        let (dst_vk_format, dst_is_rgb) = to_vk_format_info(dst.format)?;

        if vk_format != dst_vk_format || src_is_rgb != dst_is_rgb {
            return Err(ScalixError::ExecutionFailed(format!(
                "LoD Downscaler format conversion not supported directly: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let is_profiling = self.profiler.is_enabled();
        let t0_wall = if is_profiling { Some(std::time::Instant::now()) } else { None };

        let vk_filter = to_vk_filter(options.filter);
        let device = &self.ctx.device;

        // Branchless single-cycle hardware intrinsic calculation using ilog2
        let ratio = if dst.width == 0 || dst.height == 0 {
            1
        } else {
            (src.width / dst.width).min(src.height / dst.height)
        };
        let num_levels = if ratio > 1 {
            let calculated = ratio.ilog2().saturating_add(1);
            if options.vulkan.max_mip_levels > 0 {
                calculated.min(options.vulkan.max_mip_levels)
            } else {
                calculated
            }
        } else {
            1
        };

        let src_size = if src_is_rgb {
            (src.width as usize * src.height as usize * 4) as vk::DeviceSize
        } else {
            src.data.len() as vk::DeviceSize
        };

        let dst_size = if dst_is_rgb {
            (dst.width as usize * dst.height as usize * 4) as vk::DeviceSize
        } else {
            dst.data.len() as vk::DeviceSize
        };

        unsafe {
            // 1. Create Staging Buffers via RAII guards
            let src_staging = GpuBuffer::allocate(
                &self.ctx,
                src_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let src_staging_buf = src_staging.buffer;
            let src_staging_mem = src_staging.memory;

            // Copy source data into staging memory (unpack RGB -> RGBA if needed)
            let t_unpack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let mapped_src = device
                .map_memory(src_staging_mem, 0, src_size, vk::MemoryMapFlags::empty())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to map src staging memory: {e}")))?
                as *mut u8;

            if src_is_rgb {
                let pixel_count = (src.width * src.height) as usize;
                let src_raw = src.data.as_ptr();
                for i in 0..pixel_count {
                    let s = src_raw.add(i * 3);
                    let d = mapped_src.add(i * 4);
                    *d = *s;
                    *d.add(1) = *s.add(1);
                    *d.add(2) = *s.add(2);
                    *d.add(3) = 255;
                }
            } else {
                std::ptr::copy_nonoverlapping(src.data.as_ptr(), mapped_src, src.data.len());
            }
            device.unmap_memory(src_staging_mem);
            let host_unpack_ms = t_unpack_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            // 2. Create Destination Staging Buffer
            let dst_staging = GpuBuffer::allocate(
                &self.ctx,
                dst_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let dst_staging_buf = dst_staging.buffer;
            let dst_staging_mem = dst_staging.memory;

            // 3. Create Multi-Level Source VkImage (Mip Pyramid Allocation)
            let src_gpu_img = GpuImage::allocate(
                &self.ctx,
                vk_format,
                src.width,
                src.height,
                num_levels,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            let src_image = src_gpu_img.image;

            // 4. Create Single-Level Destination VkImage
            let dst_gpu_img = GpuImage::allocate(
                &self.ctx,
                dst_vk_format,
                dst.width,
                dst.height,
                1,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            let dst_image = dst_gpu_img.image;

            // 5. Allocate and Record Command Buffer
            let cmd_guard = CommandBufferGuard::allocate(&self.ctx)?;
            let cmd_buf = cmd_guard.cmd_buf;

            let query_guard = if is_profiling {
                QueryPoolGuard::new(&self.ctx, 4)
            } else {
                None
            };
            let query_pool = query_guard.as_ref().map(|q| q.pool);

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to begin command buffer: {e}")))?;

            if let Some(qp) = query_pool {
                device.cmd_reset_query_pool(cmd_buf, qp, 0, 4);
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            }

            // A. Transition Level 0 to TRANSFER_DST_OPTIMAL
            let barrier_base = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: src_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_base],
            );

            // B. Upload Host Staging Buffer to Level 0
            let copy_region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: src.width,
                    height: src.height,
                    depth: 1,
                },
            };
            device.cmd_copy_buffer_to_image(
                cmd_buf,
                src_staging_buf,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TRANSFER, qp, 1);
            }

            // C. Generate Hierarchical Mip Pyramid (Level 0 -> Level 1 -> ... -> Level N-1)
            let mut mip_w = src.width as i32;
            let mut mip_h = src.height as i32;

            for i in 0..(num_levels - 1) {
                let next_mip_w = (mip_w / 2).max(1);
                let next_mip_h = (mip_h / 2).max(1);

                // Transition level i from TRANSFER_DST_OPTIMAL to TRANSFER_SRC_OPTIMAL
                let barrier_src_level = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: src_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    ..Default::default()
                };

                // Transition level i+1 from UNDEFINED to TRANSFER_DST_OPTIMAL
                let barrier_dst_level = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: src_image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i + 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                };

                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_src_level, barrier_dst_level],
                );

                // 2x2 Box Filter reduction via hardware blit
                let blit_mip = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: mip_w, y: mip_h, z: 1 },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i + 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D { x: next_mip_w, y: next_mip_h, z: 1 },
                    ],
                };

                device.cmd_blit_image(
                    cmd_buf,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit_mip],
                    vk::Filter::LINEAR,
                );

                mip_w = next_mip_w;
                mip_h = next_mip_h;
            }

            // Transition the last mip level (num_levels - 1) to TRANSFER_SRC_OPTIMAL
            let last_level_barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: src_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: num_levels - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[last_level_barrier],
            );

            // D. Select optimal pre-filtered mip level for final destination resolve
            let chosen_level = num_levels - 1;
            let src_mip_w = (src.width >> chosen_level).max(1) as i32;
            let src_mip_h = (src.height >> chosen_level).max(1) as i32;

            // E. Transition Destination Image to TRANSFER_DST_OPTIMAL
            let barrier_dst_init = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: dst_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_dst_init],
            );

            // F. Final Hardware Resolve Blit from chosen_level -> dst_image
            let final_blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: chosen_level,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: src_mip_w, y: src_mip_h, z: 1 },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: dst.width as i32, y: dst.height as i32, z: 1 },
                ],
            };

            device.cmd_blit_image(
                cmd_buf,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[final_blit],
                vk_filter,
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TRANSFER, qp, 2);
            }

            // G. Transition dst_image to TRANSFER_SRC_OPTIMAL for readback
            let barrier_dst_readback = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: dst_image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_dst_readback],
            );

            // H. Copy Destination Image to Staging Buffer
            let copy_dst_region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: dst.width,
                    height: dst.height,
                    depth: 1,
                },
            };
            device.cmd_copy_image_to_buffer(
                cmd_buf,
                dst_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_staging_buf,
                &[copy_dst_region],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TRANSFER, qp, 3);
            }

            device
                .end_command_buffer(cmd_buf)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to end command buffer: {e}")))?;

            // 7. Submit Queue and Synchronize
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            let t_sync_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            device
                .queue_submit(self.ctx.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to submit queue: {e}")))?;

            device
                .queue_wait_idle(self.ctx.queue)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Queue wait idle failed: {e}")))?;
            let driver_sync_ms = t_sync_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let mut gpu_upload_ms = 0.0;
            let mut gpu_pure_blit_ms = 0.0;
            let mut gpu_download_ms = 0.0;

            // Retrieve GPU timestamp results if query pool active
            if let Some(qp) = query_pool {
                let mut timestamps = [0u64; 4];
                if device.get_query_pool_results(
                    qp,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                ).is_ok() {
                    let period_ms = (self.ctx.timestamp_period as f64) * 1e-6;
                    gpu_upload_ms = timestamps[1].saturating_sub(timestamps[0]) as f64 * period_ms;
                    gpu_pure_blit_ms = timestamps[2].saturating_sub(timestamps[1]) as f64 * period_ms;
                    gpu_download_ms = timestamps[3].saturating_sub(timestamps[2]) as f64 * period_ms;
                }
            }

            // 8. Copy Back from Destination Staging Memory (pack RGBA -> RGB if needed)
            let t_repack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let mapped_dst = device
                .map_memory(dst_staging_mem, 0, dst_size, vk::MemoryMapFlags::empty())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to map dst staging memory: {e}")))?
                as *const u8;

            if dst_is_rgb {
                let pixel_count = (dst.width * dst.height) as usize;
                let dst_raw = dst.data.as_mut_ptr();
                for i in 0..pixel_count {
                    let s = mapped_dst.add(i * 4);
                    let d = dst_raw.add(i * 3);
                    *d = *s;
                    *d.add(1) = *s.add(1);
                    *d.add(2) = *s.add(2);
                }
            } else {
                std::ptr::copy_nonoverlapping(mapped_dst, dst.data.as_mut_ptr(), dst.data.len());
            }
            device.unmap_memory(dst_staging_mem);
            let host_repack_ms = t_repack_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            if is_profiling {
                let total_wall_ms = t0_wall.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                self.profiler.record(ProfileMetrics {
                    host_unpack_ms,
                    gpu_upload_ms,
                    gpu_pure_blit_ms,
                    gpu_download_ms,
                    host_repack_ms,
                    driver_sync_ms,
                    total_wall_ms,
                });
            }

            Ok(())
        }
    }
}

impl VulkanPipeline for VulkanLodDownscaler {
    #[inline]
    fn name(&self) -> &'static str {
        "VulkanLodDownscaler"
    }

    #[inline]
    fn strategy(&self) -> VulkanStrategy {
        VulkanStrategy::LodPyramid
    }

    #[inline]
    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, options: &ResizeOptions) -> Result<()> {
        self.process(src, dst, options)
    }
}
