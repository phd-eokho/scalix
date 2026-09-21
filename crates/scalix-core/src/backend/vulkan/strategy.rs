//! Vulkan Pipeline Strategy Hierarchy

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum VulkanStrategy {
    Auto = 0,
    /// Hardware Blitter (`vkCmdBlitImage`) using fixed-function GPU 2D scaling units.
    Blit = 1,
    /// Offscreen Raster Graphics pipeline (`vkCmdDraw` fullscreen quad/triangle with hardware sampler).
    Raster = 2,
    /// Hierarchical LoD / Mipchain Downscaler (multi-pass pyramid reduction).
    LodPyramid = 3,
    /// Compute Shader Kernel (`vkCmdDispatch` with programmable filters).
    Compute = 4,
}

impl Default for VulkanStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Vulkan-specific execution metadata and options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct VulkanOptions {
    pub strategy: VulkanStrategy,
    /// Maximum mipmap levels to generate during hierarchical LoD downscaling.
    /// Set to 0 for automatic (generates all levels down to destination size).
    pub max_mip_levels: u32,
}
