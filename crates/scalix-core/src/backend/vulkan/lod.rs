//! Hierarchical LoD / Mipchain Downscaler Pipeline
//!
//! Generates multi-pass mip pyramid chains (by factors of 2x2) to eliminate
//! aliasing and shimmering during extreme downscaling (e.g. 4K -> 360p).

use std::sync::Arc;
use crate::backend::vulkan::context::VulkanContext;
use crate::types::{FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

pub struct VulkanLodDownscaler {
    #[allow(dead_code)]
    ctx: Arc<VulkanContext>,
}

impl VulkanLodDownscaler {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub fn process(&self, _src: &ImageDesc, _dst: &mut ImageDescMut, _filter: FilterMode) -> Result<()> {
        Err(ScalixError::ExecutionFailed(
            "Vulkan LoD Downscaler pipeline is currently being developed; use Blit, Raster, or Auto.".to_string(),
        ))
    }
}
