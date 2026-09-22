//! Programmable Compute Shader Resampler Pipeline (`vkCmdDispatch`)
//!
//! Programmable SPIR-V compute kernels for custom and hybrid spatial filtering.
//! Note: Deferred in favor of fixed-function and graphics hardware execution paths.

use crate::backend::vulkan::context::VulkanContext;
use crate::backend::vulkan::{VulkanPipeline, VulkanStrategy};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result, ScalixError};
use std::sync::Arc;

pub struct VulkanComputeResizer {
    #[allow(dead_code)]
    ctx: Arc<VulkanContext>,
}

impl VulkanComputeResizer {
    #[inline]
    #[must_use]
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub fn process(
        &self,
        _src: &ImageDesc,
        _dst: &mut ImageDescMut,
        _filter: FilterMode,
    ) -> Result<()> {
        Err(ScalixError::ExecutionFailed(
            "Vulkan Compute pipeline is deferred; use Blit, Raster, or Auto strategy.".to_string(),
        ))
    }
}

impl VulkanPipeline for VulkanComputeResizer {
    #[inline]
    fn name(&self) -> &'static str {
        "VulkanComputeResizer"
    }

    #[inline]
    fn strategy(&self) -> VulkanStrategy {
        VulkanStrategy::Compute
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
