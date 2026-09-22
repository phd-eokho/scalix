//! Vulkan Hardware Acceleration Backend
//!
//! Provides headless, offscreen image scaling across hardware execution paths:
//! - `VulkanBlitter`: `vkCmdBlitImage` fixed-function hardware scaling
//! - `VulkanRasterResizer`: `vkCmdDraw` fullscreen quad with hardware sampler
//! - `VulkanLodDownscaler`: Hierarchical multi-pass mip pyramid
//! - `VulkanComputeResizer`: `vkCmdDispatch` compute shader kernels

pub mod blit;
pub mod compute;
pub mod context;
pub mod lod;
pub mod raster;
pub mod rgb_compute;
pub mod ring;
pub mod strategy;
pub mod util;

pub use blit::VulkanBlitter;
pub use compute::{ComputeKernel, ComputePushConsts, VulkanComputeResizer};
pub use context::VulkanContext;
pub use lod::VulkanLodDownscaler;
pub use raster::VulkanRasterResizer;
pub use rgb_compute::VulkanRgbCompute;
pub use ring::{StagingSlot, VulkanStagingRing};
pub use strategy::{VulkanOptions, VulkanStrategy};
pub use util::{
    cpu_repack_rgb888, cpu_unpack_rgb888, CommandBufferGuard, GpuBuffer, GpuImage, QueryPoolGuard,
};

use crate::backend::Backend;
use crate::profiler::{ActiveProfiler, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result};
use std::sync::Arc;

/// Pluggable interface for Vulkan execution pipelines.
pub trait VulkanPipeline: Send + Sync {
    fn name(&self) -> &'static str;
    fn strategy(&self) -> VulkanStrategy;
    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()>;
}

pub struct VulkanBackend {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
    blitter: VulkanBlitter,
    raster: VulkanRasterResizer,
    lod: VulkanLodDownscaler,
    compute: VulkanComputeResizer,
}

impl VulkanBackend {
    /// Initializes a new Vulkan backend with a default active profiler.
    pub fn new() -> Result<Self> {
        Self::with_profiler(Arc::new(ActiveProfiler::new()))
    }

    /// Initializes a new Vulkan backend with an explicit profiler.
    pub fn with_profiler(profiler: Arc<dyn Profiler>) -> Result<Self> {
        let ctx = VulkanContext::new()?;
        let blitter = VulkanBlitter::new(Arc::clone(&ctx), Arc::clone(&profiler));
        let raster = VulkanRasterResizer::new(Arc::clone(&ctx), Arc::clone(&profiler));
        let lod = VulkanLodDownscaler::new(Arc::clone(&ctx), Arc::clone(&profiler));
        let compute = VulkanComputeResizer::new(Arc::clone(&ctx), Arc::clone(&profiler))?;

        Ok(Self {
            ctx,
            profiler,
            blitter,
            raster,
            lod,
            compute,
        })
    }

    #[inline]
    #[must_use]
    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.ctx
    }

    #[inline]
    #[must_use]
    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }

    #[inline]
    #[must_use]
    pub fn compute_resizer(&self) -> &VulkanComputeResizer {
        &self.compute
    }

    /// Registers a custom pluggable compute shader kernel for a given filter mode.
    #[inline]
    pub fn register_compute_shader(&self, filter: FilterMode, spv_bytes: &[u8]) -> Result<()> {
        self.compute.register_shader(filter, spv_bytes)
    }
}

impl Backend for VulkanBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "Vulkan Hardware Backend (Headless / Compute / Blit / Raster)"
    }

    #[inline]
    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    #[inline]
    fn is_available(&self) -> bool {
        true
    }

    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()> {
        if options.filter == FilterMode::Passthrough {
            return crate::backend::PassthroughBackend::new().process(src, dst, options);
        }

        let chosen_strategy = match options.vulkan.strategy {
            VulkanStrategy::Auto => {
                // Specialized fast-path: Direct fused compute resizer for identical packed 24-bit formats
                if src.format == dst.format
                    && matches!(
                        src.format,
                        crate::types::PixelFormat::Rgb888 | crate::types::PixelFormat::Bgr888
                    )
                {
                    VulkanStrategy::Compute
                } else {
                    // Default to hardware blitter for 1:1 format scaling
                    VulkanStrategy::Blit
                }
            }
            other => other,
        };

        match chosen_strategy {
            VulkanStrategy::Blit => self.blitter.process(src, dst, options.filter),
            VulkanStrategy::Raster => self.raster.process(src, dst, options.filter),
            VulkanStrategy::LodPyramid => self.lod.process(src, dst, options),
            VulkanStrategy::Compute => self.compute.process(src, dst, options.filter),
            VulkanStrategy::Auto => unreachable!("Auto strategy mapped prior to execution"),
        }
    }
}
