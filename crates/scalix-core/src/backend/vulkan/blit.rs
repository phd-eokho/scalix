//! Option A: Hardware Blitter Pipeline (`vkCmdBlitImage`)

use std::sync::Arc;
use ash::vk;
use crate::backend::vulkan::context::VulkanContext;
use crate::types::{FilterMode, ImageDesc, ImageDescMut, PixelFormat, Result, ScalixError};

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
        PixelFormat::Nv12 | PixelFormat::Yuv420p => {
            Err(ScalixError::UnsupportedFormat(format))
        }
    }
}

pub fn to_vk_filter(filter: FilterMode) -> vk::Filter {
    match filter {
        FilterMode::Nearest => vk::Filter::NEAREST,
        _ => vk::Filter::LINEAR,
    }
}

use crate::profiler::{ProfileMetrics, Profiler};

pub struct VulkanBlitter {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
}

impl VulkanBlitter {
    pub fn new(ctx: Arc<VulkanContext>, profiler: Arc<dyn Profiler>) -> Self {
        Self { ctx, profiler }
    }

    pub fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, filter: FilterMode) -> Result<()> {
        let (vk_format, src_is_rgb) = to_vk_format_info(src.format)?;
        let (dst_vk_format, dst_is_rgb) = to_vk_format_info(dst.format)?;

        if vk_format != dst_vk_format || src_is_rgb != dst_is_rgb {
            return Err(ScalixError::ExecutionFailed(format!(
                "Blit format conversion not supported directly: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let is_profiling = self.profiler.is_enabled();
        let t0_wall = if is_profiling { Some(std::time::Instant::now()) } else { None };

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
            // 1. Create Staging Buffers
            let src_buf_info = vk::BufferCreateInfo {
                size: src_size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let src_staging_buf = device.create_buffer(&src_buf_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create src staging buffer: {}", e))
            })?;

            let src_mem_reqs = device.get_buffer_memory_requirements(src_staging_buf);
            let src_mem_type = self.ctx.find_memory_type(
                src_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let src_alloc_info = vk::MemoryAllocateInfo {
                allocation_size: src_mem_reqs.size,
                memory_type_index: src_mem_type,
                ..Default::default()
            };
            let src_staging_mem = device.allocate_memory(&src_alloc_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to allocate src staging memory: {}", e))
            })?;
            device.bind_buffer_memory(src_staging_buf, src_staging_mem, 0).unwrap();

            // Copy source data to staging buffer (with RGB888 -> RGBA8888 unpack if necessary)
            let t_unpack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let ptr = device.map_memory(src_staging_mem, 0, src_size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
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
            let host_unpack_ms = t_unpack_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let dst_buf_info = vk::BufferCreateInfo {
                size: dst_size,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };
            let dst_staging_buf = device.create_buffer(&dst_buf_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create dst staging buffer: {}", e))
            })?;

            let dst_mem_reqs = device.get_buffer_memory_requirements(dst_staging_buf);
            let dst_mem_type = self.ctx.find_memory_type(
                dst_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let dst_alloc_info = vk::MemoryAllocateInfo {
                allocation_size: dst_mem_reqs.size,
                memory_type_index: dst_mem_type,
                ..Default::default()
            };
            let dst_staging_mem = device.allocate_memory(&dst_alloc_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to allocate dst staging memory: {}", e))
            })?;
            device.bind_buffer_memory(dst_staging_buf, dst_staging_mem, 0).unwrap();

            // 2. Create GPU Images
            let src_img_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk_format,
                extent: vk::Extent3D { width: src.width, height: src.height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let src_image = device.create_image(&src_img_info, None).unwrap();
            let src_img_mem_reqs = device.get_image_memory_requirements(src_image);
            let src_img_mem_type = self.ctx.find_memory_type(
                src_img_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let src_img_alloc = vk::MemoryAllocateInfo {
                allocation_size: src_img_mem_reqs.size,
                memory_type_index: src_img_mem_type,
                ..Default::default()
            };
            let src_img_mem = device.allocate_memory(&src_img_alloc, None).unwrap();
            device.bind_image_memory(src_image, src_img_mem, 0).unwrap();

            let dst_img_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk_format,
                extent: vk::Extent3D { width: dst.width, height: dst.height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let dst_image = device.create_image(&dst_img_info, None).unwrap();
            let dst_img_mem_reqs = device.get_image_memory_requirements(dst_image);
            let dst_img_mem_type = self.ctx.find_memory_type(
                dst_img_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let dst_img_alloc = vk::MemoryAllocateInfo {
                allocation_size: dst_img_mem_reqs.size,
                memory_type_index: dst_img_mem_type,
                ..Default::default()
            };
            let dst_img_mem = device.allocate_memory(&dst_img_alloc, None).unwrap();
            device.bind_image_memory(dst_image, dst_img_mem, 0).unwrap();

            // 3. Allocate and Record Command Buffer
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.ctx.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let cmd_buf = device.allocate_command_buffers(&alloc_info).unwrap()[0];

            let query_pool = if is_profiling {
                let query_pool_info = vk::QueryPoolCreateInfo {
                    query_type: vk::QueryType::TIMESTAMP,
                    query_count: 4,
                    ..Default::default()
                };
                device.create_query_pool(&query_pool_info, None).ok()
            } else {
                None
            };

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device.begin_command_buffer(cmd_buf, &begin_info).unwrap();

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
                image_extent: vk::Extent3D { width: src.width, height: src.height, depth: 1 },
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

            // Issue Hardware Blit (Option A)
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
                    vk::Offset3D { x: src.width as i32, y: src.height as i32, z: 1 },
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
                image_extent: vk::Extent3D { width: dst.width, height: dst.height, depth: 1 },
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

            device.end_command_buffer(cmd_buf).unwrap();

            // Submit and wait for queue execution
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            let t_sync_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            device.queue_submit(self.ctx.queue, &[submit_info], vk::Fence::null()).unwrap();
            device.queue_wait_idle(self.ctx.queue).unwrap();
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
                device.destroy_query_pool(qp, None);
            }

            // Copy result from staging buffer to destination slice (with RGBA -> RGB repack if necessary)
            let t_repack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let out_ptr = device.map_memory(dst_staging_mem, 0, dst_size, vk::MemoryMapFlags::empty()).unwrap() as *const u8;
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

            // Clean up resources
            device.free_command_buffers(self.ctx.command_pool, &[cmd_buf]);
            device.destroy_image(src_image, None);
            device.free_memory(src_img_mem, None);
            device.destroy_image(dst_image, None);
            device.free_memory(dst_img_mem, None);
            device.destroy_buffer(src_staging_buf, None);
            device.free_memory(src_staging_mem, None);
            device.destroy_buffer(dst_staging_buf, None);
            device.free_memory(dst_staging_mem, None);
            Ok(())
        }
    }
}
