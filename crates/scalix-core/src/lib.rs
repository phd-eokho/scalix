pub mod backend;
pub mod buffer;
pub mod dma;
pub mod engine;
pub mod profiler;
pub mod topology;
pub mod types;
pub mod worker;

pub use backend::gl::{
    EglContext, GlBackend, GlBlitter, GlComputeResizer, GlLodDownscaler, GlOptions,
    GlRasterResizer, GlStrategy,
};
pub use backend::opencl::{
    OpenClBackend, OpenClComputeResizer, OpenClContext, OpenClOptions, OpenClStagingRing,
    OpenClStrategy,
};
pub use backend::vulkan::{
    ComputeKernel, ComputePushConsts, VulkanBackend, VulkanComputeResizer, VulkanOptions,
    VulkanStrategy,
};
pub use backend::{Backend, PassthroughBackend};
pub use buffer::OwnedImage;
pub use dma::{DmaAllocation, DmaAllocator, DmaAllocatorType, DmaBuffer, DmaSyncFlags};
pub use engine::{Engine, EngineConfig};
pub use profiler::{ActiveProfiler, NoopProfiler, ProfileMetrics, Profiler};
pub use topology::{pci_vendors, GpuTopology};
pub use types::{
    AlignedBuffer, BackendOptions, BackendType, FilterMode, GpuDeviceKind, GpuTopologyInfo,
    ImageDesc, ImageDescMut, ImageDimensions, ImageGeometry, PixelFormat, ResizeOptions, Result,
    ScalixError, REQUIRED_MEMORY_ALIGNMENT,
};
pub use worker::{TaskHandle, WorkerPool};
