use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::bounded;

use crate::backend::{Backend, PassthroughBackend};
use crate::buffer::OwnedImage;
use crate::profiler::{ActiveProfiler, ProfileMetrics, Profiler};
use crate::types::{BackendType, ImageDesc, ImageDescMut, ResizeOptions, Result, ScalixError};
use crate::worker::{TaskHandle, WorkerPool, DEFAULT_WORKER_CAPACITY};

pub const MAX_THREAD_PREFIX_LEN: usize = 7;
pub const DEFAULT_WORKER_QUEUE_CAP: usize = DEFAULT_WORKER_CAPACITY;
pub const DEFAULT_PARALLELISM: usize = 4;
pub const DEFAULT_NUM_HW_THREADS: usize = 1;
pub const MIN_CALLBACK_WORKERS: usize = 2;

/// Configuration options for initializing the Scalix Engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub num_hw_threads: usize,
    pub num_callback_workers: usize,
    pub worker_queue_capacity: usize,
    pub thread_prefix: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(DEFAULT_PARALLELISM);
        Self {
            num_hw_threads: DEFAULT_NUM_HW_THREADS,
            num_callback_workers: cpus.max(MIN_CALLBACK_WORKERS),
            worker_queue_capacity: DEFAULT_WORKER_QUEUE_CAP,
            thread_prefix: None,
        }
    }
}

impl EngineConfig {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    #[must_use]
    pub fn with_thread_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.thread_prefix = Some(prefix.into());
        self
    }

    #[inline]
    #[must_use]
    pub fn with_hw_threads(mut self, threads: usize) -> Self {
        self.num_hw_threads = threads;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_callback_workers(mut self, workers: usize) -> Self {
        self.num_callback_workers = workers;
        self
    }

    #[inline]
    #[must_use]
    pub fn with_queue_capacity(mut self, cap: usize) -> Self {
        self.worker_queue_capacity = cap;
        self
    }
}

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
            if trimmed.len() > MAX_THREAD_PREFIX_LEN {
                let truncated = &trimmed[..MAX_THREAD_PREFIX_LEN];
                log::warn!(
                    "Thread prefix '{}' exceeds the maximum limit of {} characters and will be truncated to '{}' to guarantee compliance with Linux 15-char thread name limit.",
                    trimmed,
                    MAX_THREAD_PREFIX_LEN,
                    truncated
                );
                truncated.to_string()
            } else {
                trimmed.to_string()
            }
        }
        _ => {
            let pid = std::process::id().to_string();
            if pid.len() > MAX_THREAD_PREFIX_LEN {
                pid[..MAX_THREAD_PREFIX_LEN].to_string()
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
            BackendType::Auto => {
                match crate::backend::VulkanBackend::with_profiler(Arc::clone(&profiler)) {
                    Ok(vk_backend) => Arc::new(vk_backend),
                    Err(vk_err) => {
                        log::info!(
                            "Vulkan auto-initialization skipped ({:?}); trying OpenGL...",
                            vk_err
                        );
                        match crate::backend::GlBackend::with_profiler(Arc::clone(&profiler)) {
                            Ok(gl_backend) => Arc::new(gl_backend),
                            Err(gl_err) => {
                                log::info!(
                                    "OpenGL auto-initialization skipped ({:?}); falling back to passthrough",
                                    gl_err
                                );
                                Arc::new(PassthroughBackend::new())
                            }
                        }
                    }
                }
            }
            BackendType::Vulkan => Arc::new(crate::backend::VulkanBackend::with_profiler(
                Arc::clone(&profiler),
            )?),
            BackendType::OpenGL => Arc::new(crate::backend::GlBackend::with_profiler(Arc::clone(
                &profiler,
            ))?),
            BackendType::Passthrough | BackendType::Cpu => Arc::new(PassthroughBackend::new()),
            other => return Err(ScalixError::BackendUnavailable(other)),
        };

        let mut config = EngineConfig::default();
        if let Some(p) = prefix {
            config.thread_prefix = Some(p.to_string());
        }

        Ok(Self::with_config_and_profiler(backend, config, profiler))
    }

    /// Initializes an Engine with custom backend and configuration.
    pub fn with_config(backend: Arc<dyn Backend>, config: EngineConfig) -> Self {
        let profiler: Arc<dyn Profiler> = Arc::new(ActiveProfiler::new());
        Self::with_config_and_profiler(backend, config, profiler)
    }

    /// Initializes an Engine with custom backend, configuration, and explicit profiler.
    pub fn with_config_and_profiler(
        backend: Arc<dyn Backend>,
        config: EngineConfig,
        profiler: Arc<dyn Profiler>,
    ) -> Self {
        let p = sanitize_thread_prefix(config.thread_prefix.as_deref());
        let hw_prefix = format!("{}/scx-hw", p);
        let cb_prefix = format!("{}/scx-w", p);
        let cap = config.worker_queue_capacity;

        Self {
            backend,
            profiler,
            hw_executor: Arc::new(
                WorkerPool::try_with_prefix(
                    &hw_prefix,
                    config.num_hw_threads.max(DEFAULT_NUM_HW_THREADS),
                    cap,
                )
                .expect("Failed to initialize hardware executor worker pool"),
            ),
            callback_pool: Arc::new(
                WorkerPool::try_with_prefix(&cb_prefix, config.num_callback_workers.max(1), cap)
                    .expect("Failed to initialize callback worker pool"),
            ),
        }
    }

    /// Enables or disables profiling dynamically.
    pub fn set_profiling(&self, enabled: bool) {
        self.profiler.set_enabled(enabled);
    }

    /// Retrieves the most recent profile metrics if profiling is active.
    #[must_use]
    pub fn last_profile(&self) -> Option<ProfileMetrics> {
        self.profiler.last_metrics()
    }

    /// Returns a reference to the active profiler handle.
    #[must_use]
    pub fn profiler(&self) -> &Arc<dyn Profiler> {
        &self.profiler
    }

    /// Returns the name of the active backend provider.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Returns the backend type of the active provider.
    #[must_use]
    pub fn backend_type(&self) -> BackendType {
        self.backend.backend_type()
    }

    /// 1. Synchronous execution: blocks caller thread until resize completes.
    pub fn resize_sync<O: Into<ResizeOptions>>(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: O,
    ) -> Result<()> {
        let opt = options.into();
        self.backend.process(src, dst, &opt)
    }

    /// 2. Asynchronous execution: queues on dedicated hardware thread and returns TaskHandle.
    pub fn resize_async<O: Into<ResizeOptions>>(
        &self,
        src: OwnedImage,
        mut dst: OwnedImage,
        options: O,
    ) -> TaskHandle<OwnedImage> {
        let opt = options.into();
        let (tx, rx) = bounded(1);
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let backend = Arc::clone(&self.backend);

        let submit_res = self.hw_executor.submit(move || {
            let res = backend
                .process(&src.as_desc(), &mut dst.as_desc_mut(), &opt)
                .map(|_| dst);
            let _ = tx.send(res);
            done_flag.store(true, Ordering::Release);
        });

        if let Err(e) = submit_res {
            let (err_tx, err_rx) = bounded(1);
            let _ = err_tx.send(Err(e));
            done.store(true, Ordering::Release);
            return TaskHandle::new(err_rx, done);
        }

        TaskHandle::new(rx, done)
    }

    /// Asynchronous execution with raw pointer descriptors (zero-copy, no buffer cloning).
    ///
    /// # Safety
    /// Caller must ensure that pointers referenced in `src` and `dst` remain valid until the returned `TaskHandle` completes.
    pub unsafe fn resize_async_raw<O: Into<ResizeOptions>>(
        &self,
        src: ImageDesc<'static>,
        mut dst: ImageDescMut<'static>,
        options: O,
    ) -> TaskHandle<()> {
        let opt = options.into();
        let (tx, rx) = bounded(1);
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let backend = Arc::clone(&self.backend);

        let submit_res = self.hw_executor.submit(move || {
            let res = backend.process(&src, &mut dst, &opt);
            let _ = tx.send(res);
            done_flag.store(true, Ordering::Release);
        });

        if let Err(e) = submit_res {
            let (err_tx, err_rx) = bounded(1);
            let _ = err_tx.send(Err(e));
            done.store(true, Ordering::Release);
            return TaskHandle::new(err_rx, done);
        }

        TaskHandle::new(rx, done)
    }

    /// 3. Callback-driven execution:
    ///
    /// Hardware kernel executes on dedicated HW thread, then immediately dispatches
    /// the user callback / post-processing to the multi-worker thread pool.
    pub fn resize_callback<O: Into<ResizeOptions>, F>(
        &self,
        src: OwnedImage,
        mut dst: OwnedImage,
        options: O,
        callback: F,
    ) -> Result<()>
    where
        F: FnOnce(Result<OwnedImage>) + Send + 'static,
    {
        let opt = options.into();
        let backend = Arc::clone(&self.backend);
        let callback_pool = Arc::clone(&self.callback_pool);

        self.hw_executor.submit(move || {
            let res = backend
                .process(&src.as_desc(), &mut dst.as_desc_mut(), &opt)
                .map(|_| dst);

            // Offload callback & post-processing to the multi-worker callback pool,
            // freeing the HW executor thread to immediately process the next frame!
            let _ = callback_pool.submit(move || {
                callback(res);
            });
        })
    }
}
