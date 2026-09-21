//! Programmable Compute Shader Resampler Pipeline (`vkCmdDispatch`)
//!
//! Programmable SPIR-V compute kernels for custom and hybrid spatial filtering.
//! Note: Deferred in favor of fixed-function and graphics hardware execution paths.

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
        Err(ScalixError::ExecutionFailed(
            "Vulkan Compute pipeline is deferred; use Blit, Raster, or Auto strategy.".to_string(),
        ))
    }
}
