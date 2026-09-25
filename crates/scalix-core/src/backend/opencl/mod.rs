//! OpenCL Hardware Acceleration Backend
//!
//! Provides headless, cross-platform OpenCL compute scaling:
//! - `OpenClComputeResizer`: `clEnqueueNDRangeKernel` direct buffer resampling

pub mod compute;
pub mod context;
pub mod ring;
pub mod strategy;

pub use compute::OpenClComputeResizer;
pub use context::OpenClContext;
pub use ring::OpenClStagingRing;
pub use strategy::{OpenClOptions, OpenClStrategy};

use crate::backend::Backend;
use crate::profiler::{ActiveProfiler, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result};
use std::sync::Arc;

pub struct OpenClBackend {
    ctx: Arc<OpenClContext>,
    profiler: Arc<dyn Profiler>,
    compute: OpenClComputeResizer,
}

impl OpenClBackend {
    pub fn new() -> Result<Self> {
        Self::with_profiler(Arc::new(ActiveProfiler::new()))
    }

    pub fn with_profiler(profiler: Arc<dyn Profiler>) -> Result<Self> {
        let ctx = OpenClContext::new()?;
        let compute = OpenClComputeResizer::new(Arc::clone(&ctx), Arc::clone(&profiler))?;
        Ok(Self {
            ctx,
            profiler,
            compute,
        })
    }

    #[inline]
    #[must_use]
    pub fn context(&self) -> &Arc<OpenClContext> {
        &self.ctx
    }

    #[inline]
    #[must_use]
    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }
}

impl Backend for OpenClBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "OpenCL Hardware Backend (Direct Compute)"
    }

    #[inline]
    fn backend_type(&self) -> BackendType {
        BackendType::OpenCL
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

        self.compute.process(src, dst, options.filter)
    }
}
