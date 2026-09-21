//! Option C: Offscreen Raster Graphics Pipeline (`vkCmdDraw` with hardware sampler)
//!
//! Utilizes GPU texture sampling hardware (bilinear, trilinear, anisotropic filtering)
//! via fullscreen triangle rendering into offscreen framebuffers.

use std::sync::Arc;
use crate::backend::vulkan::context::VulkanContext;
use crate::types::{FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

pub struct VulkanRasterResizer {
    #[allow(dead_code)]
    ctx: Arc<VulkanContext>,
}

impl VulkanRasterResizer {
    pub fn new(ctx: Arc<VulkanContext>) -> Self {
        Self { ctx }
    }

    pub fn process(&self, _src: &ImageDesc, _dst: &mut ImageDescMut, _filter: FilterMode) -> Result<()> {
        // Scaffolded for Option C expansion:
        // 1. Fullscreen triangle vertex shader (3 vertices via gl_VertexIndex).
        // 2. Fragment shader sampling combined image sampler.
        // 3. RenderPass + Framebuffer render target.
        Err(ScalixError::ExecutionFailed(
            "Vulkan Raster pipeline (Option C) is currently being scaffolded; use Option A Blit or Auto.".to_string(),
        ))
    }
}
