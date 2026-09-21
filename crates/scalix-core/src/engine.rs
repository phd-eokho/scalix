use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::backend::{Backend, PassthroughBackend};
use crate::buffer::OwnedImage;
use crate::profiler::{ActiveProfiler, ProfileMetrics, Profiler};
use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};
use crate::worker::{TaskHandle, WorkerPool};

/// Unified Scalix Engine coordinator.
///
/// Designed with a two-tier execution pipeline:
/// - `hw_executor`: Dedicated single-threaded executor maintaining hardware context affinity (e.g., EGL / Vulkan).
/// - `callback_pool`: Multi-threaded worker pool executing callbacks and user post-processing in parallel.
pub struct Engine {
    backend: Arc<dyn Backend>,
    profiler: Arc<dyn Profiler>,
    hw_executor: Arc<WorkerPool>,
    callback_pool: Arc<WorkerPool>,
}

/// Sanitizes and limits the thread prefix to ensure names fit within Linux's 15-character limit.
/// Suffixes: `/scx-hw` (7 chars) and `/scx-w<id>` (7-8 chars).
/// Maximum allowed prefix length is 7 characters.
pub fn sanitize_thread_prefix(custom_prefix: Option<&str>) -> String {
    match custom_prefix {
        Some(p) if !p.trim().is_empty() => {
            let trimmed = p.trim();
            if trimmed.len() > 7 {
                let truncated = &trimmed[..7];
                log::warn!(
                    "Thread prefix '{}' exceeds the maximum limit of 7 characters and will be truncated to '{}' to guarantee compliance with Linux 15-char thread name limit.",
                    trimmed,
                    truncated
                );
                truncated.to_string()
            } else {
                trimmed.to_string()
            }
        }
        _ => {
            let pid = std::process::id().to_string();
            if pid.len() > 7 {
                pid[..7].to_string()
            } else {
                pid
            }
        }
    }
}

impl Engine {
    /// Initializes a new Scalix Engine with the requested backend and default PID thread prefix (`<pid>/scx-hw`).
    pub fn new(backend_type: BackendType) -> Result<Self> {
        Self::with_prefix(backend_type, None)
    }

    /// Initializes a new Scalix Engine with the requested backend and optional custom thread prefix.
    pub fn with_prefix(backend_type: BackendType, prefix: Option<&str>) -> Result<Self> {
        let profiler: Arc<dyn Profiler> = Arc::new(ActiveProfiler::new());
        Self::with_profiler(backend_type, prefix, profiler)
    }

    /// Initializes a new Scalix Engine with explicit profiler.
    pub fn with_profiler(
        backend_type: BackendType,
        prefix: Option<&str>,
        profiler: Arc<dyn Profiler>,
    ) -> Result<Self> {
        let backend: Arc<dyn Backend> = match backend_type {
            BackendType::Auto => match crate::backend::VulkanBackend::with_profiler(Arc::clone(&profiler)) {
                Ok(vk_backend) => Arc::new(vk_backend),
                Err(e) => {
                    log::info!(
                        "Vulkan auto-initialization skipped ({:?}); falling back to passthrough",
                        e
                    );
                    Arc::new(PassthroughBackend::new())
                }
            },
            BackendType::Vulkan => {
                Arc::new(crate::backend::VulkanBackend::with_profiler(Arc::clone(&profiler))?)
            }
            BackendType::Passthrough | BackendType::Cpu => Arc::new(PassthroughBackend::new()),
            other => return Err(ScalixError::BackendUnavailable(other)),
        };

        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let p = sanitize_thread_prefix(prefix);
        let hw_prefix = format!("{}/scx-hw", p);
        let cb_prefix = format!("{}/scx-w", p);

        // 1 dedicated hardware context thread, multi-worker callback pool
        let hw_executor = Arc::new(WorkerPool::with_prefix(&hw_prefix, 1));
        let callback_pool = Arc::new(WorkerPool::with_prefix(&cb_prefix, num_cpus.max(2)));

        Ok(Self {
            backend,
            profiler,
            hw_executor,
            callback_pool,
        })
    }

    /// Enables or disables profiling dynamically.
    pub fn set_profiling(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }

    /// Retrieves the most recent profile metrics if profiling is active.
    pub fn last_profile(&self) -> Option<ProfileMetrics> {
        self.profiler.last_metrics()
    }

    /// Returns a reference to the active profiler handle.
    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }

    /// Initializes an Engine with custom backend, configurable thread concurrency, and optional prefix.
    pub fn with_config(
        backend: Arc<dyn Backend>,
        num_hw_threads: usize,
        num_callback_workers: usize,
        prefix: Option<&str>,
    ) -> Self {
        let profiler: Arc<dyn Profiler> = Arc::new(ActiveProfiler::new());
        let p = sanitize_thread_prefix(prefix);
        let hw_prefix = format!("{}/scx-hw", p);
        let cb_prefix = format!("{}/scx-w", p);

        Self {
            backend,
            profiler,
            hw_executor: Arc::new(WorkerPool::with_prefix(
                &hw_prefix,
                num_hw_threads.max(1),
            )),
            callback_pool: Arc::new(WorkerPool::with_prefix(
                &cb_prefix,
                num_callback_workers.max(1),
            )),
        }
    }

    /// Returns the name of the active backend provider.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Returns the backend type of the active provider.
    pub fn backend_type(&self) -> BackendType {
        self.backend.backend_type()
    }

    /// 1. Synchronous execution: blocks caller thread until resize completes.
    pub fn resize_sync(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        self.backend.process(src, dst, filter)
    }

    /// 2. Asynchronous execution: queues on dedicated hardware thread and returns TaskHandle.
    pub fn resize_async(
        &self,
        src: OwnedImage,
        mut dst: OwnedImage,
        filter: FilterMode,
    ) -> TaskHandle<OwnedImage> {
        let (tx, rx) = mpsc::channel();
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let backend = Arc::clone(&self.backend);

        let submit_res = self.hw_executor.submit(move || {
            let res = backend
                .process(&src.as_desc(), &mut dst.as_desc_mut(), filter)
                .map(|_| dst);
            done_flag.store(true, Ordering::Release);
            let _ = tx.send(res);
        });

        if let Err(e) = submit_res {
            done.store(true, Ordering::Release);
            let (err_tx, err_rx) = mpsc::channel();
            let _ = err_tx.send(Err(e));
            return TaskHandle::new(err_rx, done);
        }

        TaskHandle::new(rx, done)
    }

    /// 3. Callback-driven execution:
    /// Hardware kernel executes on dedicated HW thread, then immediately dispatches
    /// the user callback / post-processing to the multi-worker thread pool.
    pub fn resize_callback<F>(
        &self,
        src: OwnedImage,
        mut dst: OwnedImage,
        filter: FilterMode,
        callback: F,
    ) -> Result<()>
    where
        F: FnOnce(Result<OwnedImage>) + Send + 'static,
    {
        let backend = Arc::clone(&self.backend);
        let callback_pool = Arc::clone(&self.callback_pool);

        self.hw_executor.submit(move || {
            let res = backend
                .process(&src.as_desc(), &mut dst.as_desc_mut(), filter)
                .map(|_| dst);

            // Offload callback & post-processing to the multi-worker callback pool,
            // freeing the HW executor thread to immediately process the next frame!
            let _ = callback_pool.submit(move || {
                callback(res);
            });
        })
    }
}
