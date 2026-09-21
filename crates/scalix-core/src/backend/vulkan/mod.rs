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
pub mod strategy;

pub use blit::VulkanBlitter;
pub use compute::VulkanComputeResizer;
pub use context::VulkanContext;
pub use lod::VulkanLodDownscaler;
pub use raster::VulkanRasterResizer;
pub use strategy::{VulkanOptions, VulkanStrategy};

use std::sync::Arc;
use crate::backend::Backend;
use crate::profiler::{ActiveProfiler, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result};

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
        let compute = VulkanComputeResizer::new(Arc::clone(&ctx));

        Ok(Self {
            ctx,
            profiler,
            blitter,
            raster,
            lod,
            compute,
        })
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.ctx
    }

    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        "Vulkan Hardware Backend (Headless / Compute / Blit / Raster)"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn is_available(&self) -> bool {
        true
    }

    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, options: &ResizeOptions) -> Result<()> {
        if options.filter == FilterMode::Passthrough {
            let copy_len = src.data.len().min(dst.data.len());
            dst.data[..copy_len].copy_from_slice(&src.data[..copy_len]);
            return Ok(());
        }

        let chosen_strategy = match options.vulkan.strategy {
            VulkanStrategy::Auto => {
                // Default to hardware blitter for 1:1 format scaling
                VulkanStrategy::Blit
            }
            other => other,
        };

        match chosen_strategy {
            VulkanStrategy::Blit | VulkanStrategy::Auto => {
                self.blitter.process(src, dst, options.filter)
            }
            VulkanStrategy::Raster => {
                self.raster.process(src, dst, options.filter)
            }
            VulkanStrategy::LodPyramid => {
                self.lod.process(src, dst, options)
            }
            VulkanStrategy::Compute => {
                self.compute.process(src, dst, options.filter)
            }
        }
    }
}
