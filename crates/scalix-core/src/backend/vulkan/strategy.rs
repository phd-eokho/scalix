//! Vulkan Pipeline Strategy Hierarchy (HW-oriented: A -> C -> D -> B)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanStrategy {
    /// Option A: Hardware Blitter (`vkCmdBlitImage`) using fixed-function GPU scaling units.
    Blit,
    /// Option C: Offscreen Raster Graphics pipeline (`vkCmdDraw` fullscreen triangle with hardware sampler).
    Raster,
    /// Option D: Hierarchical LoD / Mipchain Downscaler (multi-pass pyramid reduction).
    LodPyramid,
    /// Option B: Compute Shader Kernel (`vkCmdDispatch` with programmable filters like Lanczos3/Bicubic).
    Compute,
    /// Automatic selection based on hardware capabilities and filter mode.
    Auto,
}

impl Default for VulkanStrategy {
    fn default() -> Self {
        Self::Auto
    }
}
