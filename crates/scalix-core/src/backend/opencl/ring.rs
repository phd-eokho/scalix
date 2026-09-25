//! OpenCL Staging Ring Buffer and Memory Slot Manager
//!
//! Provides ring-buffered `cl_mem` device allocations and `cl_event` tracking to
//! eliminate per-frame buffer creation/destruction overhead and enable asynchronous pipelining.

use std::sync::Arc;

use crate::backend::opencl::context::*;
use crate::types::{Result, ScalixError};

pub const DEFAULT_OPENCL_RING_SLOTS: usize = 3;

/// Trailing padding bytes to prevent word-boundary overrun.
pub const BUFFER_TAIL_PADDING_BYTES: usize = 16;

/// An isolated in-flight execution slot in the OpenCL staging ring.
pub struct OpenClStagingSlot {
    /// Cached source memory buffer (CPU -> GPU upload)
    pub src_buffer: ClMem,
    pub src_buf_size: usize,

    /// Cached destination memory buffer (GPU compute output)
    pub dst_buffer: ClMem,
    pub dst_buf_size: usize,

    /// Optional completion event for in-flight command tracking
    pub completion_event: Option<ClEvent>,
}

unsafe impl Send for OpenClStagingSlot {}
unsafe impl Sync for OpenClStagingSlot {}

impl Default for OpenClStagingSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenClStagingSlot {
    pub fn new() -> Self {
        Self {
            src_buffer: std::ptr::null_mut(),
            src_buf_size: 0,
            dst_buffer: std::ptr::null_mut(),
            dst_buf_size: 0,
            completion_event: None,
        }
    }

    /// Waits for any in-flight execution associated with this slot and releases the event.
    pub fn wait_and_clear_event(&mut self, ctx: &Arc<OpenClContext>) -> Result<()> {
        if let Some(ev) = self.completion_event.take() {
            unsafe {
                let err = (ctx.cl.clWaitForEvents)(1, &ev);
                (ctx.cl.clReleaseEvent)(ev);
                if err != CL_SUCCESS {
                    return Err(ScalixError::ExecutionFailed(format!(
                        "OpenCL clWaitForEvents failed: {}",
                        err
                    )));
                }
            }
        }
        Ok(())
    }

    /// Ensures that the slot has a source `cl_mem` buffer with at least `min_size` bytes.
    pub fn ensure_src_buffer(
        &mut self,
        ctx: &Arc<OpenClContext>,
        min_size: usize,
    ) -> Result<ClMem> {
        let needed_size = min_size.saturating_add(BUFFER_TAIL_PADDING_BYTES);
        let needs_realloc = self.src_buffer.is_null()
            || self.src_buf_size < needed_size
            || needed_size < self.src_buf_size / 2;

        if needs_realloc {
            if !self.src_buffer.is_null() {
                unsafe { (ctx.cl.clReleaseMemObject)(self.src_buffer) };
                self.src_buffer = std::ptr::null_mut();
            }

            let alloc_size = needed_size.max(64 * 1024);
            let mut err: ClInt = 0;
            let buf = unsafe {
                (ctx.cl.clCreateBuffer)(
                    ctx.context,
                    CL_MEM_READ_ONLY,
                    alloc_size,
                    std::ptr::null_mut(),
                    &mut err,
                )
            };
            if err != CL_SUCCESS || buf.is_null() {
                return Err(ScalixError::ExecutionFailed(format!(
                    "Failed to allocate OpenCL source buffer (size {}): err={}",
                    alloc_size, err
                )));
            }
            self.src_buffer = buf;
            self.src_buf_size = alloc_size;
        }

        Ok(self.src_buffer)
    }

    /// Ensures that the slot has a destination `cl_mem` buffer with at least `min_size` bytes.
    pub fn ensure_dst_buffer(
        &mut self,
        ctx: &Arc<OpenClContext>,
        min_size: usize,
    ) -> Result<ClMem> {
        let needed_size = min_size.saturating_add(BUFFER_TAIL_PADDING_BYTES);
        let needs_realloc = self.dst_buffer.is_null()
            || self.dst_buf_size < needed_size
            || needed_size < self.dst_buf_size / 2;

        if needs_realloc {
            if !self.dst_buffer.is_null() {
                unsafe { (ctx.cl.clReleaseMemObject)(self.dst_buffer) };
                self.dst_buffer = std::ptr::null_mut();
            }

            let alloc_size = needed_size.max(64 * 1024);
            let mut err: ClInt = 0;
            let buf = unsafe {
                (ctx.cl.clCreateBuffer)(
                    ctx.context,
                    CL_MEM_READ_WRITE,
                    alloc_size,
                    std::ptr::null_mut(),
                    &mut err,
                )
            };
            if err != CL_SUCCESS || buf.is_null() {
                return Err(ScalixError::ExecutionFailed(format!(
                    "Failed to allocate OpenCL destination buffer (size {}): err={}",
                    alloc_size, err
                )));
            }
            self.dst_buffer = buf;
            self.dst_buf_size = alloc_size;
        }

        Ok(self.dst_buffer)
    }

    pub fn destroy(&mut self, ctx: &Arc<OpenClContext>) {
        let _ = self.wait_and_clear_event(ctx);
        unsafe {
            if !self.src_buffer.is_null() {
                (ctx.cl.clReleaseMemObject)(self.src_buffer);
                self.src_buffer = std::ptr::null_mut();
            }
            if !self.dst_buffer.is_null() {
                (ctx.cl.clReleaseMemObject)(self.dst_buffer);
                self.dst_buffer = std::ptr::null_mut();
            }
        }
    }
}

pub struct OpenClStagingRing {
    ctx: Arc<OpenClContext>,
    slots: Vec<OpenClStagingSlot>,
    current_slot: usize,
}

unsafe impl Send for OpenClStagingRing {}
unsafe impl Sync for OpenClStagingRing {}

impl OpenClStagingRing {
    pub fn new(ctx: Arc<OpenClContext>, slot_count: usize) -> Result<Self> {
        let count = slot_count.max(1);
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            slots.push(OpenClStagingSlot::new());
        }
        Ok(Self {
            ctx,
            slots,
            current_slot: 0,
        })
    }

    /// Acquires the next available staging slot, advancing the ring index.
    pub fn acquire_slot(&mut self) -> Result<(usize, &mut OpenClStagingSlot)> {
        let idx = self.current_slot;
        self.current_slot = (self.current_slot + 1) % self.slots.len();
        let slot = &mut self.slots[idx];
        slot.wait_and_clear_event(&self.ctx)?;
        Ok((idx, slot))
    }
}

impl Drop for OpenClStagingRing {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            slot.destroy(&self.ctx);
        }
    }
}
