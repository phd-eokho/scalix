//! Hardware Blitter Pipeline (`vkCmdBlitImage`)

use crate::backend::vulkan::context::VulkanContext;
use crate::types::{
    FilterMode, ImageDesc, ImageDescMut, PixelFormat, ResizeOptions, Result, ScalixError,
};
use ash::vk;
use std::sync::Arc;

pub fn to_vk_format_info(format: PixelFormat) -> Result<(vk::Format, bool)> {
    match format {
        PixelFormat::Rgba8888 => Ok((vk::Format::R8G8B8A8_UNORM, false)),
        PixelFormat::Bgra8888 => Ok((vk::Format::B8G8R8A8_UNORM, false)),
        PixelFormat::Rgb888 => Ok((vk::Format::R8G8B8A8_UNORM, true)),
        PixelFormat::Bgr888 => Ok((vk::Format::B8G8R8A8_UNORM, true)),
        PixelFormat::R8 => Ok((vk::Format::R8_UNORM, false)),
        PixelFormat::Rg88 => Ok((vk::Format::R8G8_UNORM, false)),
        PixelFormat::Rgba16f => Ok((vk::Format::R16G16B16A16_SFLOAT, false)),
        PixelFormat::Rgba32f => Ok((vk::Format::R32G32B32A32_SFLOAT, false)),
        PixelFormat::Nv12 | PixelFormat::Yuv420p => Err(ScalixError::UnsupportedFormat(format)),
    }
}

pub fn to_vk_filter(filter: FilterMode) -> vk::Filter {
    match filter {
        FilterMode::Nearest => vk::Filter::NEAREST,
        _ => vk::Filter::LINEAR,
    }
}

use crate::backend::vulkan::util::{CommandBufferGuard, GpuBuffer, GpuImage, QueryPoolGuard};
use crate::backend::vulkan::{VulkanPipeline, VulkanStrategy};
use crate::profiler::{ProfileMetrics, Profiler};

pub struct VulkanBlitter {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
}

impl VulkanBlitter {
    #[inline]
    #[must_use]
    pub fn new(ctx: Arc<VulkanContext>, profiler: Arc<dyn Profiler>) -> Self {
        Self { ctx, profiler }
    }

    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        let (vk_format, src_is_rgb) = to_vk_format_info(src.format)?;
        let (dst_vk_format, dst_is_rgb) = to_vk_format_info(dst.format)?;

        if vk_format != dst_vk_format || src_is_rgb != dst_is_rgb {
            return Err(ScalixError::ExecutionFailed(format!(
                "Blit format conversion not supported directly: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let is_profiling = self.profiler.is_enabled();
        let t0_wall = if is_profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let vk_filter = to_vk_filter(filter);
        let device = &self.ctx.device;

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

            // Copy source data to staging buffer (with RGB888 -> RGBA8888 unpack if necessary)
            let t_unpack_start = if is_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let ptr = device
                .map_memory(src_staging_mem, 0, src_size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to map src staging memory: {e}"))
                })? as *mut u8;
            if src_is_rgb {
                let num_pixels = (src.width * src.height) as usize;
                for p in 0..num_pixels {
                    let s_idx = p * 3;
                    let d_idx = p * 4;
                    *ptr.add(d_idx) = src.data[s_idx];
                    *ptr.add(d_idx + 1) = src.data[s_idx + 1];
                    *ptr.add(d_idx + 2) = src.data[s_idx + 2];
                    *ptr.add(d_idx + 3) = 255;
                }
            } else {
                std::ptr::copy_nonoverlapping(src.data.as_ptr(), ptr, src.data.len());
            }
            device.unmap_memory(src_staging_mem);
            let host_unpack_ms = t_unpack_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let dst_staging = GpuBuffer::allocate(
                &self.ctx,
                dst_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let dst_staging_buf = dst_staging.buffer;
            let dst_staging_mem = dst_staging.memory;
            // 2. Create GPU Images via RAII guards
            let src_gpu_img = GpuImage::allocate(
                &self.ctx,
                vk_format,
                src.width,
                src.height,
                1,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            let src_image = src_gpu_img.image;

            let dst_gpu_img = GpuImage::allocate(
                &self.ctx,
                vk_format,
                dst.width,
                dst.height,
                1,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            let dst_image = dst_gpu_img.image;

            // 3. Allocate and Record Command Buffer via RAII guard
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
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to begin command buffer: {e}"))
                })?;

            if let Some(qp) = query_pool {
                device.cmd_reset_query_pool(cmd_buf, qp, 0, 4);
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            }

            // Transition src_image UNDEFINED -> TRANSFER_DST_OPTIMAL
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            let barrier_to_dst = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image: src_image,
                subresource_range,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_dst],
            );

            // Copy staging buffer to src_image
            let buffer_image_copy = vk::BufferImageCopy {
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
                &[buffer_image_copy],
            );

            // Transition src_image TRANSFER_DST_OPTIMAL -> TRANSFER_SRC_OPTIMAL
            let barrier_src_blit = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                image: src_image,
                subresource_range,
                ..Default::default()
            };

            // Transition dst_image UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier_dst_blit = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image: dst_image,
                subresource_range,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src_blit, barrier_dst_blit],
            );

            // Issue Hardware Blit
            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 1);
            }

            let blit_region = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src.width as i32,
                        y: src.height as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst.width as i32,
                        y: dst.height as i32,
                        z: 1,
                    },
                ],
            };

            device.cmd_blit_image(
                cmd_buf,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk_filter,
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 2);
            }

            // Transition dst_image TRANSFER_DST_OPTIMAL -> TRANSFER_SRC_OPTIMAL for readback
            let barrier_dst_readback = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                image: dst_image,
                subresource_range,
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

            // Copy dst_image back to dst_staging_buf
            let dst_buffer_image_copy = vk::BufferImageCopy {
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
                &[dst_buffer_image_copy],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 3);
            }

            device.end_command_buffer(cmd_buf).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to end command buffer: {e}"))
            })?;

            // Submit and wait for queue execution
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            let t_sync_start = if is_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            device
                .queue_submit(self.ctx.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to submit queue: {e}"))
                })?;
            device.queue_wait_idle(self.ctx.queue).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to wait for queue idle: {e}"))
            })?;
            let driver_sync_ms = t_sync_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            let mut gpu_upload_ms = 0.0;
            let mut gpu_pure_blit_ms = 0.0;
            let mut gpu_download_ms = 0.0;

            // Retrieve GPU timestamp results if query pool active
            if let Some(qp) = query_pool {
                let mut timestamps = [0u64; 4];
                if device
                    .get_query_pool_results(
                        qp,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .is_ok()
                {
                    let period_ms = (self.ctx.timestamp_period as f64) * 1e-6;
                    gpu_upload_ms = timestamps[1].saturating_sub(timestamps[0]) as f64 * period_ms;
                    gpu_pure_blit_ms =
                        timestamps[2].saturating_sub(timestamps[1]) as f64 * period_ms;
                    gpu_download_ms =
                        timestamps[3].saturating_sub(timestamps[2]) as f64 * period_ms;
                }
            }

            // Copy result from staging buffer to destination slice (with RGBA -> RGB repack if necessary)
            let t_repack_start = if is_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let out_ptr = device
                .map_memory(dst_staging_mem, 0, dst_size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to map dst staging memory: {e}"))
                })? as *const u8;
            if dst_is_rgb {
                let num_pixels = (dst.width * dst.height) as usize;
                for p in 0..num_pixels {
                    let s_idx = p * 4;
                    let d_idx = p * 3;
                    dst.data[d_idx] = *out_ptr.add(s_idx);
                    dst.data[d_idx + 1] = *out_ptr.add(s_idx + 1);
                    dst.data[d_idx + 2] = *out_ptr.add(s_idx + 2);
                }
            } else {
                std::ptr::copy_nonoverlapping(out_ptr, dst.data.as_mut_ptr(), dst.data.len());
            }
            device.unmap_memory(dst_staging_mem);
            let host_repack_ms = t_repack_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            if is_profiling {
                let total_wall_ms = t0_wall
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
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

impl VulkanPipeline for VulkanBlitter {
    #[inline]
    fn name(&self) -> &'static str {
        "VulkanBlitter"
    }

    #[inline]
    fn strategy(&self) -> VulkanStrategy {
        VulkanStrategy::Blit
    }

    #[inline]
    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()> {
        self.process(src, dst, options.filter)
    }
}
