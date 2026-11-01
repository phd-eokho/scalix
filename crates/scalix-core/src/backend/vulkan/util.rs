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
    pub usage: vk::BufferUsageFlags,
    ctx: Arc<VulkanContext>,
}

impl GpuBuffer {
    #[inline]
    pub fn new(
        ctx: Arc<VulkanContext>,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self {
            buffer,
            memory,
            size,
            usage,
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
        let is_device_local = properties.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let is_host_visible = properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        if usage.contains(vk::BufferUsageFlags::TRANSFER_DST) && !is_host_visible {
            ctx.topology
                .validate_writeback_memory(is_device_local, is_host_visible);
        }

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
            usage,
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

/// RAII owner of a `vk::CommandPool`; destroying the pool frees its command buffers.
///
/// Vulkan requires a pool and all command buffers allocated from it to be externally
/// synchronized, so every independently locked owner (each staging ring, each LoD call) gets its
/// own pool instead of sharing one across threads.
pub struct CommandPool {
    pub pool: vk::CommandPool,
    ctx: Arc<VulkanContext>,
}

impl CommandPool {
    pub fn new(ctx: &Arc<VulkanContext>, flags: vk::CommandPoolCreateFlags) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo {
            queue_family_index: ctx.queue_family_index,
            flags,
            ..Default::default()
        };
        let pool = unsafe {
            ctx.device.create_command_pool(&info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create command pool: {e}"))
            })?
        };
        Ok(Self {
            pool,
            ctx: Arc::clone(ctx),
        })
    }

    pub fn allocate(&self, count: u32) -> Result<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: self.pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: count,
            ..Default::default()
        };
        unsafe {
            self.ctx
                .device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to allocate command buffers: {e}"))
                })
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A one-shot primary command buffer backed by its own transient pool.
pub struct CommandBufferGuard {
    pub cmd_buf: vk::CommandBuffer,
    _pool: CommandPool,
}

impl CommandBufferGuard {
    pub fn allocate(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let pool = CommandPool::new(ctx, vk::CommandPoolCreateFlags::TRANSIENT)?;
        let cmd_buf = pool.allocate(1)?.into_iter().next().ok_or_else(|| {
            ScalixError::ExecutionFailed("Allocated command buffer is empty".to_string())
        })?;
        Ok(Self {
            cmd_buf,
            _pool: pool,
        })
    }
}

/// Descriptor sets one resize can allocate: an RGB unpack, the main pass and an RGB repack.
const RESIZE_MAX_DESCRIPTOR_SETS: u32 = 3;
/// Per-type descriptor budget covering the bindings of those three passes.
const RESIZE_MAX_DESCRIPTORS_PER_TYPE: u32 = 4;

/// RAII owner of a reusable `vk::DescriptorPool` sized for one resize.
///
/// Owners reset it once the previous resize's GPU work has completed instead of creating and
/// destroying a pool per pass; destroying the pool frees its sets.
pub struct DescriptorPool {
    pub pool: vk::DescriptorPool,
    ctx: Arc<VulkanContext>,
}

impl DescriptorPool {
    pub fn for_resize(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let pool_sizes = [
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ]
        .map(|ty| vk::DescriptorPoolSize {
            ty,
            descriptor_count: RESIZE_MAX_DESCRIPTORS_PER_TYPE,
        });
        let info = vk::DescriptorPoolCreateInfo {
            max_sets: RESIZE_MAX_DESCRIPTOR_SETS,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe {
            ctx.device
                .create_descriptor_pool(&info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to create descriptor pool: {e}"))
                })?
        };
        Ok(Self {
            pool,
            ctx: Arc::clone(ctx),
        })
    }

    pub fn allocate(&self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            ..Default::default()
        };
        let sets = unsafe {
            self.ctx
                .device
                .allocate_descriptor_sets(&info)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to allocate descriptor set: {e}"))
                })?
        };
        sets.into_iter().next().ok_or_else(|| {
            ScalixError::ExecutionFailed("Allocated descriptor set list is empty".to_string())
        })
    }

    /// Frees every set; the GPU must no longer use any of them.
    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.ctx
                .device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to reset descriptor pool: {e}"))
                })
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// RAII Drop guard for a `vk::Fence`.
pub struct FenceGuard {
    pub fence: vk::Fence,
    ctx: Arc<VulkanContext>,
}

impl FenceGuard {
    pub fn new(ctx: &Arc<VulkanContext>) -> Result<Self> {
        let fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create fence: {e}")))?
        };
        Ok(Self {
            fence,
            ctx: Arc::clone(ctx),
        })
    }
}

impl Drop for FenceGuard {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_fence(self.fence, None);
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
    if num_pixels == 0 {
        return;
    }

    debug_assert_eq!(
        dst as usize % 4,
        0,
        "Destination buffer pointer must be at least 4-byte aligned"
    );

    let src_ptr = src.as_ptr();
    let dst_ptr = dst as *mut u32;

    unsafe {
        let bulk_pixels = num_pixels - 1;
        for p in 0..bulk_pixels {
            let s_p = src_ptr.add(p * 3);
            let w = (s_p as *const u32).read_unaligned();
            let px = (w & 0x00FF_FFFF) | 0xFF00_0000;
            dst_ptr.add(p).write(px);
        }

        // Final pixel: read 3 bytes explicitly without reading past src buffer
        let last_p = num_pixels - 1;
        let s_last = src_ptr.add(last_p * 3);
        let r = *s_last as u32;
        let g = (*s_last.add(1) as u32) << 8;
        let b = (*s_last.add(2) as u32) << 16;
        let px = r | g | b | 0xFF00_0000;
        dst_ptr.add(last_p).write(px);
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
