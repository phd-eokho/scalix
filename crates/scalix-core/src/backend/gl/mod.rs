//! OpenGL / GLES Hardware Acceleration Backend
//!
//! Provides headless, offscreen image scaling using EGL and Framebuffer Objects (FBO):
//! - `GlBlitter`: `glBlitFramebuffer` 2D hardware scaling
//! - `GlRasterResizer`: Offscreen fullscreen quad draw with bilinear/bicubic shaders
//! - `GlLodDownscaler`: Hierarchical multi-pass mipchain pyramid
//! - `GlComputeResizer`: `glDispatchCompute` compute shader pipeline

pub mod blit;
pub mod compute;
pub mod context;
pub mod lod;
pub mod raster;
pub mod ring;
pub mod strategy;

pub use blit::GlBlitter;
pub use compute::GlComputeResizer;
pub use context::EglContext;
pub use lod::GlLodDownscaler;
pub use raster::GlRasterResizer;
pub use ring::{GlStagingRing, DEFAULT_GL_RING_SLOTS};
pub use strategy::{GlOptions, GlStrategy};

use crate::backend::Backend;
use crate::profiler::{ActiveProfiler, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result};
use std::sync::{Arc, Mutex};

pub struct GlBackend {
    ctx: Arc<EglContext>,
    ring: Arc<Mutex<GlStagingRing>>,
    profiler: Arc<dyn Profiler>,
    blitter: GlBlitter,
    raster: GlRasterResizer,
    lod: GlLodDownscaler,
    compute: GlComputeResizer,
}

impl GlBackend {
    /// Initializes a new OpenGL backend with a default active profiler.
    pub fn new() -> Result<Self> {
        Self::with_profiler(Arc::new(ActiveProfiler::new()))
    }

    /// Initializes a new OpenGL backend with an explicit profiler.
    pub fn with_profiler(profiler: Arc<dyn Profiler>) -> Result<Self> {
        let ctx = Arc::new(EglContext::new()?);
        let ring = Arc::new(Mutex::new(GlStagingRing::new(
            Arc::clone(&ctx),
            DEFAULT_GL_RING_SLOTS,
        )?));

        let blitter = GlBlitter::new(Arc::clone(&ctx), Arc::clone(&ring), Arc::clone(&profiler));
        let raster =
            GlRasterResizer::new(Arc::clone(&ctx), Arc::clone(&ring), Arc::clone(&profiler));
        let lod = GlLodDownscaler::new(Arc::clone(&ctx), Arc::clone(&ring), Arc::clone(&profiler));
        let compute =
            GlComputeResizer::new(Arc::clone(&ctx), Arc::clone(&ring), Arc::clone(&profiler));

        Ok(Self {
            ctx,
            ring,
            profiler,
            blitter,
            raster,
            lod,
            compute,
        })
    }

    #[inline]
    #[must_use]
    pub fn context(&self) -> &Arc<EglContext> {
        &self.ctx
    }

    #[inline]
    #[must_use]
    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }

    #[inline]
    #[must_use]
    pub fn staging_ring(&self) -> &Arc<Mutex<GlStagingRing>> {
        &self.ring
    }
}

impl Backend for GlBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "OpenGL/GLES Hardware Backend (Headless EGL / FBO Blit / Raster / Compute / LoD)"
    }

    #[inline]
    fn backend_type(&self) -> BackendType {
        BackendType::OpenGL
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

        let gl_opts = options.gl_options();
        let chosen_strategy = match gl_opts.strategy {
            GlStrategy::Auto => {
                if matches!(
                    options.filter,
                    FilterMode::Bicubic | FilterMode::Lanczos3 | FilterMode::Area
                ) {
                    GlStrategy::Raster
                } else {
                    GlStrategy::Blit
                }
            }
            other => other,
        };

        match chosen_strategy {
            GlStrategy::Blit => self.blitter.process(src, dst, options.filter),
            GlStrategy::Raster => self.raster.process(src, dst, options.filter),
            GlStrategy::LodPyramid => self.lod.process(src, dst, options),
            GlStrategy::Compute => self.compute.process(src, dst, options.filter),
            GlStrategy::Auto => self.blitter.process(src, dst, options.filter),
        }
    }
}
