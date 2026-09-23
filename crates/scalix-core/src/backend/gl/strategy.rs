//! OpenGL/GLES backend strategy definitions

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub enum GlStrategy {
    #[default]
    Auto = 0,
    Blit = 1,
    Raster = 2,
    LodPyramid = 3,
    Compute = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct GlOptions {
    pub strategy: GlStrategy,
    pub max_mip_levels: u32,
}

impl Default for GlOptions {
    fn default() -> Self {
        Self {
            strategy: GlStrategy::Auto,
            max_mip_levels: 0,
        }
    }
}
