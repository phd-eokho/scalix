pub mod gl;
pub mod passthrough;
pub mod vulkan;

pub use gl::{GlBackend, GlOptions, GlStrategy};
pub use passthrough::PassthroughBackend;
pub use vulkan::{VulkanBackend, VulkanOptions, VulkanPipeline, VulkanStrategy};

use crate::types::{BackendType, ImageDesc, ImageDescMut, ResizeOptions, Result};

/// Common hardware/software backend provider interface.
pub trait Backend: Send + Sync {
    /// Returns human-readable name of the backend.
    fn name(&self) -> &'static str;

    /// Returns the backend type enumeration.
    fn backend_type(&self) -> BackendType;

    /// Checks if this backend is supported and initialized on the current system.
    fn is_available(&self) -> bool;

    /// Executes image processing / resize from source to destination with dynamic options.
    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()>;
}
