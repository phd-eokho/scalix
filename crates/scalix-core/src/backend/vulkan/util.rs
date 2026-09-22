//! Vulkan RAII Resource Guards and Helper Allocators
//!
//! Provides drop guards for GPU buffers, device memories, images, views, command buffers,
//! and query pools to prevent GPU resource leaks on early returns.

use crate::backend::vulkan::context::VulkanContext;
use crate::types::{Result, ScalixError};
use ash::vk;
use std::sync::Arc;

/// RAII Drop guard encapsulating a `vk::Buffer` and its bound `vk::DeviceMemory`.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    ctx: Arc<VulkanContext>,
}

impl GpuBuffer {
    #[inline]
    pub fn new(
        ctx: Arc<VulkanContext>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
    ) -> Self {
        Self {
            buffer,
            memory,
            size,
            ctx,
        }
    }

    /// Allocates and binds a GPU buffer.
    pub fn allocate(
        ctx: &Arc<VulkanContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let device = &ctx.device;
        let buf_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            device.create_buffer(&buf_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create GPU buffer: {e}"))
            })?
        };

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_type = match ctx.find_memory_type(mem_reqs.memory_type_bits, properties) {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
            ..Default::default()
        };

        let memory = unsafe {
            match device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(ScalixError::ExecutionFailed(format!(
                        "Failed to allocate GPU buffer memory: {e}"
                    )));
                }
            }
        };

        if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to bind GPU buffer memory: {e}"
            )));
        }

        Ok(Self {
            buffer,
            memory,
            size,
            ctx: Arc::clone(ctx),
        })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_buffer(self.buffer, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

/// RAII Drop guard encapsulating a `vk::Image`, its bound `vk::DeviceMemory`,
/// and an optional `vk::ImageView`.
pub struct GpuImage {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: Option<vk::ImageView>,
    ctx: Arc<VulkanContext>,
}

impl GpuImage {
    #[inline]
    pub fn new(
        ctx: Arc<VulkanContext>,
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: Option<vk::ImageView>,
    ) -> Self {
        Self {
            image,
            memory,
            view,
            ctx,
        }
    }

    /// Allocates and binds a 2D GPU image with DEVICE_LOCAL memory.
    pub fn allocate(
        ctx: &Arc<VulkanContext>,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let device = &ctx.device;
        let img_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe {
            device.create_image(&img_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create GPU image: {e}"))
            })?
        };

        let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
        let mem_type = match ctx.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) {
            Ok(m) => m,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e);
            }
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
            ..Default::default()
        };

        let memory = unsafe {
            match device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_image(image, None);
                    return Err(ScalixError::ExecutionFailed(format!(
                        "Failed to allocate GPU image memory: {e}"
                    )));
                }
            }
        };

        if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(ScalixError::ExecutionFailed(format!(
                "Failed to bind GPU image memory: {e}"
            )));
        }

        Ok(Self {
            image,
            memory,
            view: None,
            ctx: Arc::clone(ctx),
        })
    }

    /// Creates and attaches an image view to this GPU image.
    pub fn create_view(
        &mut self,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let device = &self.ctx.device;
        let view_info = vk::ImageViewCreateInfo {
            image: self.image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let view = unsafe {
            device.create_image_view(&view_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create GPU image view: {e}"))
            })?
        };

        self.view = Some(view);
        Ok(view)
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe {
            if let Some(view) = self.view {
                self.ctx.device.destroy_image_view(view, None);
            }
            self.ctx.device.destroy_image(self.image, None);
            self.ctx.device.free_memory(self.memory, None);
        }
    }
}

/// RAII Drop guard for an allocated primary command buffer.
pub struct CommandBufferGuard {
    pub cmd_buf: vk::CommandBuffer,
    pool: vk::CommandPool,
    ctx: Arc<VulkanContext>,
}

impl CommandBufferGuard {
    pub fn allocate(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: ctx.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };

        let cmd_bufs = unsafe {
            ctx.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to allocate command buffer: {e}"))
                })?
        };

        let cmd_buf = cmd_bufs.into_iter().next().ok_or_else(|| {
            ScalixError::ExecutionFailed("Allocated command buffer is empty".to_string())
        })?;

        Ok(Self {
            cmd_buf,
            pool: ctx.command_pool,
            ctx: Arc::clone(ctx),
        })
    }
}

impl Drop for CommandBufferGuard {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .free_command_buffers(self.pool, &[self.cmd_buf]);
        }
    }
}

/// RAII Drop guard for a Vulkan timestamp query pool.
pub struct QueryPoolGuard {
    pub pool: vk::QueryPool,
    ctx: Arc<VulkanContext>,
}

impl QueryPoolGuard {
    pub fn new(ctx: &Arc<VulkanContext>, count: u32) -> Option<Self> {
        let info = vk::QueryPoolCreateInfo {
            query_type: vk::QueryType::TIMESTAMP,
            query_count: count,
            ..Default::default()
        };

        let pool = unsafe { ctx.device.create_query_pool(&info, None).ok()? };
        Some(Self {
            pool,
            ctx: Arc::clone(ctx),
        })
    }
}

impl Drop for QueryPoolGuard {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_query_pool(self.pool, None);
        }
    }
}

/// Fast host CPU fallback unpack from packed RGB888 (3 bytes/px) to RGBA8888 (4 bytes/px, alpha=255).
/// Reads a 4-byte 32-bit word for each pixel (unaligned load) and masks out the next pixel's red byte
/// with alpha 255 (`(w & 0x00FF_FFFF) | 0xFF00_0000`).
/// Destination pointer `dst` MUST be at least 4-byte aligned (64-byte alignment enforced by Engine).
#[inline]
pub fn cpu_unpack_rgb888(src: &[u8], dst: *mut u8, num_pixels: usize) {
    debug_assert_eq!(
        dst as usize % 4,
        0,
        "Destination buffer pointer must be at least 4-byte aligned"
    );

    let src_ptr = src.as_ptr();
    let dst_ptr = dst as *mut u32;

    unsafe {
        for p in 0..num_pixels {
            let s_p = src_ptr.add(p * 3);
            let w = (s_p as *const u32).read_unaligned();
            let px = (w & 0x00FF_FFFF) | 0xFF00_0000;
            dst_ptr.add(p).write(px);
        }
    }
}

/// Fast host CPU fallback repack from RGBA8888 (4 bytes/px) to packed RGB888 (3 bytes/px).
/// Directly writes the 4-byte 32-bit word from aligned `src` into `dst` for each in-order pixel
/// (the 4th alpha byte is cleanly overwritten by the next pixel's red byte), writing 3 bytes
/// for the final pixel.
/// Source pointer `src` MUST be at least 4-byte aligned (64-byte alignment enforced by Engine).
#[inline]
pub fn cpu_repack_rgb888(src: *const u8, dst: &mut [u8], num_pixels: usize) {
    if num_pixels == 0 {
        return;
    }

    debug_assert_eq!(
        src as usize % 4,
        0,
        "Source buffer pointer must be at least 4-byte aligned"
    );

    let src_ptr = src as *const u32;
    let dst_ptr = dst.as_mut_ptr();

    unsafe {
        let bulk_pixels = num_pixels - 1;
        for p in 0..bulk_pixels {
            let px = *src_ptr.add(p);
            (dst_ptr.add(p * 3) as *mut u32).write_unaligned(px);
        }

        // Final pixel: write 3 bytes without overflowing dst buffer
        let last_p = num_pixels - 1;
        let px = *src_ptr.add(last_p);
        let d_p = dst_ptr.add(last_p * 3);
        *d_p = (px & 0xFF) as u8;
        *d_p.add(1) = ((px >> 8) & 0xFF) as u8;
        *d_p.add(2) = ((px >> 16) & 0xFF) as u8;
    }
}
