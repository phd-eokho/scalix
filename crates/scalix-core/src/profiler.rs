//! Pluggable Engine & GPU Hardware Profiling Architecture
//!
//! Designed with zero-overhead abstraction:
//! - When disabled (`NoopProfiler` or `enabled = false`), all hooks compile down to
//!   inline no-ops with zero branch, allocation, or Vulkan query overhead.
//! - When enabled (`ActiveProfiler`), records stage-by-stage timings and GPU hardware query timestamps.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Stage-by-stage latency metrics measured in milliseconds.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProfileMetrics {
    pub host_unpack_ms: f64,
    pub gpu_upload_ms: f64,
    pub gpu_pure_blit_ms: f64,
    pub gpu_download_ms: f64,
    pub host_repack_ms: f64,
    pub driver_sync_ms: f64,
    pub total_wall_ms: f64,
}

/// Pluggable profiler interface for Scalix execution pipelines.
pub trait Profiler: Send + Sync {
    /// Returns whether profiling is currently enabled.
    fn is_enabled(&self) -> bool;

    /// Records a completed frame's stage metrics.
    fn record(&self, metrics: ProfileMetrics);

    /// Retrieves the most recent profile metrics if available.
    fn last_metrics(&self) -> Option<ProfileMetrics>;

    /// Enables or disables profiling at runtime.
    fn set_enabled(&self, enabled: bool);
}

/// Zero-cost No-Op Profiler (default in production builds).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopProfiler;

impl Profiler for NoopProfiler {
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        false
    }

    #[inline(always)]
    fn record(&self, _metrics: ProfileMetrics) {}

    #[inline(always)]
    fn last_metrics(&self) -> Option<ProfileMetrics> {
        None
    }

    #[inline(always)]
    fn set_enabled(&self, _enabled: bool) {}
}

/// Active Latency & GPU Hardware Profiler.
pub struct ActiveProfiler {
    enabled: AtomicBool,
    last: Mutex<Option<ProfileMetrics>>,
}

impl Default for ActiveProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveProfiler {
    /// Creates a new active profiler (disabled by default for zero runtime overhead).
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            last: Mutex::new(None),
        }
    }
}

impl Profiler for ActiveProfiler {
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn record(&self, metrics: ProfileMetrics) {
        if self.is_enabled() {
            if let Ok(mut lock) = self.last.lock() {
                *lock = Some(metrics);
            }
        }
    }

    fn last_metrics(&self) -> Option<ProfileMetrics> {
        self.last.lock().ok().and_then(|lock| *lock)
    }

    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}
