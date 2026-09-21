//! Vulkan Hardware Acceleration Backend
//!
//! Provides headless, offscreen image scaling across four hardware execution paths:
//! - Option A: `VulkanBlitter` (`vkCmdBlitImage` fixed-function hardware scaling)
//! - Option C: `VulkanRasterResizer` (`vkCmdDraw` fullscreen triangle with hardware sampler)
//! - Option D: `VulkanLodDownscaler` (Hierarchical multi-pass mip pyramid)
//! - Option B: `VulkanComputeResizer` (`vkCmdDispatch` programmable SPIR-V compute kernels)

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
pub use strategy::VulkanStrategy;

use std::sync::Arc;
use crate::backend::Backend;
use crate::profiler::{ActiveProfiler, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, Result};

pub struct VulkanBackend {
    ctx: Arc<VulkanContext>,
    strategy: VulkanStrategy,
    profiler: Arc<dyn Profiler>,
    blitter: VulkanBlitter,
    raster: VulkanRasterResizer,
    lod: VulkanLodDownscaler,
    compute: VulkanComputeResizer,
}

impl VulkanBackend {
    /// Initializes a new Vulkan backend with the default (Auto) strategy and no-op/inactive profiler.
    pub fn new() -> Result<Self> {
        Self::with_strategy(VulkanStrategy::Auto)
    }

    /// Initializes a new Vulkan backend with an explicit pipeline execution strategy.
    pub fn with_strategy(strategy: VulkanStrategy) -> Result<Self> {
        Self::with_strategy_and_profiler(strategy, Arc::new(ActiveProfiler::new()))
    }

    /// Initializes a new Vulkan backend with an explicit profiler.
    pub fn with_profiler(profiler: Arc<dyn Profiler>) -> Result<Self> {
        Self::with_strategy_and_profiler(VulkanStrategy::Auto, profiler)
    }

    /// Initializes a new Vulkan backend with explicit strategy and profiler.
    pub fn with_strategy_and_profiler(
        strategy: VulkanStrategy,
        profiler: Arc<dyn Profiler>,
    ) -> Result<Self> {
        let ctx = VulkanContext::new()?;
        let blitter = VulkanBlitter::new(Arc::clone(&ctx), Arc::clone(&profiler));
        let raster = VulkanRasterResizer::new(Arc::clone(&ctx));
        let lod = VulkanLodDownscaler::new(Arc::clone(&ctx));
        let compute = VulkanComputeResizer::new(Arc::clone(&ctx));

        Ok(Self {
            ctx,
            strategy,
            profiler,
            blitter,
            raster,
            lod,
            compute,
        })
    }

    pub fn strategy(&self) -> VulkanStrategy {
        self.strategy
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
        "Vulkan Hardware Backend (Headless / Compute / Blit)"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Vulkan
    }

    fn is_available(&self) -> bool {
        true
    }

    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, filter: FilterMode) -> Result<()> {
        if filter == FilterMode::Passthrough {
            let copy_len = src.data.len().min(dst.data.len());
            dst.data[..copy_len].copy_from_slice(&src.data[..copy_len]);
            return Ok(());
        }

        let chosen_strategy = match self.strategy {
            VulkanStrategy::Auto => {
                // In Phase 3 initial milestone: Default to Option A (Hardware Blitter)
                VulkanStrategy::Blit
            }
            other => other,
        };

        match chosen_strategy {
            VulkanStrategy::Blit | VulkanStrategy::Auto => {
                self.blitter.process(src, dst, filter)
            }
            VulkanStrategy::Raster => {
                self.raster.process(src, dst, filter)
            }
            VulkanStrategy::LodPyramid => {
                self.lod.process(src, dst, filter)
            }
            VulkanStrategy::Compute => {
                self.compute.process(src, dst, filter)
            }
        }
    }
}
