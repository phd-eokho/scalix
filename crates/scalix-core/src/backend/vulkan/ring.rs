//! Vulkan Staging Ring Buffer & Pipeline Resource Manager
//!
//! Provides ring-buffered staging buffer caching, CommandBuffer reuse, and
//! per-slot VkFence synchronization to eliminate per-frame allocations and enable
//! CPU/GPU pipelining.

use crate::backend::vulkan::context::VulkanContext;
use crate::backend::vulkan::util::GpuBuffer;
use crate::types::{Result, ScalixError};
use ash::vk;
use std::sync::Arc;

pub const DEFAULT_RING_SLOTS: usize = 3;

/// An isolated in-flight execution slot in the Vulkan staging ring.
pub struct StagingSlot {
    pub cmd_buf: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub query_pool: Option<vk::QueryPool>,
    pub in_flight: bool,
    pub src_staging: Option<GpuBuffer>,
    pub dst_staging: Option<GpuBuffer>,
}

impl StagingSlot {
    /// Ensures that the slot has a source staging buffer with at least `min_size` bytes.
    ///
    /// Reuses the existing buffer if `b.size >= min_size`. To preserve system memory stability,
    /// reallocates and shrinks the buffer if the requirement drops below 1/2 of the allocated capacity.
    pub fn ensure_src_staging(
        &mut self,
        ctx: &Arc<VulkanContext>,
        min_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<&GpuBuffer> {
        let needs_realloc = match &self.src_staging {
            Some(b) => b.size < min_size || min_size < b.size / 2,
            None => true,
        };

        if needs_realloc {
            let new_size = min_size.max(64 * 1024);
            let buf = GpuBuffer::allocate(
                ctx,
                new_size,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.src_staging = Some(buf);
        }
        Ok(self.src_staging.as_ref().unwrap())
    }

    /// Ensures that the slot has a destination staging buffer with at least `min_size` bytes.
    ///
    /// Reuses the existing buffer if `b.size >= min_size`. To preserve system memory stability,
    /// reallocates and shrinks the buffer if the requirement drops below 1/2 of the allocated capacity.
    pub fn ensure_dst_staging(
        &mut self,
        ctx: &Arc<VulkanContext>,
        min_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<&GpuBuffer> {
        let needs_realloc = match &self.dst_staging {
            Some(b) => b.size < min_size || min_size < b.size / 2,
            None => true,
        };

        if needs_realloc {
            let new_size = min_size.max(64 * 1024);
            let buf = GpuBuffer::allocate(
                ctx,
                new_size,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.dst_staging = Some(buf);
        }
        Ok(self.dst_staging.as_ref().unwrap())
    }

    /// Waits for any in-flight GPU workload on this slot and resets the fence.
    pub fn wait_and_reset(&mut self, device: &ash::Device) -> Result<()> {
        if self.in_flight {
            unsafe {
                device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| {
                        ScalixError::ExecutionFailed(format!("Failed to wait for slot fence: {e}"))
                    })?;
                device.reset_fences(&[self.fence]).map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to reset slot fence: {e}"))
                })?;
            }
            self.in_flight = false;
        }
        Ok(())
    }
}

/// Ring-buffered staging allocator and execution coordinator.
pub struct VulkanStagingRing {
    ctx: Arc<VulkanContext>,
    slots: Vec<StagingSlot>,
    current_idx: usize,
}

impl VulkanStagingRing {
    /// Creates a new staging ring with `slot_count` slots (default: 3 for triple buffering).
    pub fn new(ctx: Arc<VulkanContext>, slot_count: usize) -> Result<Self> {
        let count = slot_count.max(1);
        let device = &ctx.device;

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: ctx.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: count as u32,
            ..Default::default()
        };

        let cmd_bufs = unsafe {
            device.allocate_command_buffers(&alloc_info).map_err(|e| {
                ScalixError::ExecutionFailed(format!(
                    "Failed to allocate ring command buffers: {e}"
                ))
            })?
        };

        let mut slots = Vec::with_capacity(count);
        for cmd_buf in cmd_bufs {
            let fence_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::empty(),
                ..Default::default()
            };
            let fence = unsafe {
                device.create_fence(&fence_info, None).map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to create slot fence: {e}"))
                })?
            };

            let qp_info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: 4,
                ..Default::default()
            };
            let query_pool = unsafe { device.create_query_pool(&qp_info, None).ok() };

            slots.push(StagingSlot {
                cmd_buf,
                fence,
                query_pool,
                in_flight: false,
                src_staging: None,
                dst_staging: None,
            });
        }

        Ok(Self {
            ctx,
            slots,
            current_idx: 0,
        })
    }

    /// Acquires the next available staging slot in the ring, ensuring previous work on it is complete.
    pub fn acquire_slot(&mut self) -> Result<(usize, &mut StagingSlot)> {
        let idx = self.current_idx;
        self.current_idx = (self.current_idx + 1) % self.slots.len();
        let slot = &mut self.slots[idx];
        slot.wait_and_reset(&self.ctx.device)?;
        Ok((idx, slot))
    }

    /// Flushes all pending in-flight slots across the ring.
    pub fn flush_all(&mut self) -> Result<()> {
        for slot in &mut self.slots {
            slot.wait_and_reset(&self.ctx.device)?;
        }
        Ok(())
    }
}

impl Drop for VulkanStagingRing {
    fn drop(&mut self) {
        let _ = self.flush_all();
        let device = &self.ctx.device;
        for slot in &mut self.slots {
            unsafe {
                if let Some(qp) = slot.query_pool.take() {
                    device.destroy_query_pool(qp, None);
                }
                device.destroy_fence(slot.fence, None);
                device.free_command_buffers(self.ctx.command_pool, &[slot.cmd_buf]);
            }
        }
    }
}
