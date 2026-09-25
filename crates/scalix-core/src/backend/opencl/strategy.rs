//! OpenCL backend strategy definitions

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub enum OpenClStrategy {
    #[default]
    Auto = 0,
    Compute = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct OpenClOptions {
    pub strategy: OpenClStrategy,
}

impl Default for OpenClOptions {
    fn default() -> Self {
        Self {
            strategy: OpenClStrategy::Auto,
        }
    }
}
