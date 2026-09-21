//! Option B: Programmable Compute Shader Resampler Pipeline (`vkCmdDispatch`)
//!
//! Custom SPIR-V compute kernels for high-order spatial resampling filters
//! (Lanczos3, Bicubic Catmull-Rom, Area Box, AI super-resolution).

use std::sync::Arc;
use crate::backend::vulkan::context::VulkanContext;
use crate::types::{FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

pub struct VulkanComputeResizer {
    #[allow(dead_code)]
    ctx: Arc<VulkanContext>,
}

impl VulkanComputeResizer {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub fn process(&self, _src: &ImageDesc, _dst: &mut ImageDescMut, _filter: FilterMode) -> Result<()> {
        // Scaffolded for Option B expansion:
        // 1. 2D workgroups (16x16) compute shader dispatch.
        // 2. Push constants for scale ratios and filter weights.
        // 3. Separable horizontal and vertical 1D passes with shared memory tiles.
        Err(ScalixError::ExecutionFailed(
            "Vulkan Compute pipeline (Option B) is currently being scaffolded; use Option A Blit or Auto.".to_string(),
        ))
    }
}
