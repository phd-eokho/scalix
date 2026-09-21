pub mod backend;
pub mod buffer;
pub mod dma;
pub mod engine;
pub mod profiler;
pub mod types;
pub mod worker;

pub use backend::vulkan::{VulkanBackend, VulkanOptions, VulkanStrategy};
pub use backend::{Backend, PassthroughBackend};
pub use buffer::OwnedImage;
pub use dma::{DmaBuffer, DmaSyncFlags};
pub use engine::Engine;
pub use profiler::{ActiveProfiler, NoopProfiler, ProfileMetrics, Profiler};
pub use types::{
    BackendType, FilterMode, ImageDesc, ImageDescMut, PixelFormat, ResizeOptions, Result,
    ScalixError,
};
pub use worker::{TaskHandle, WorkerPool};


