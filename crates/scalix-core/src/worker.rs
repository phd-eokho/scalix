use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};

use crate::types::{Result, ScalixError};

pub const DEFAULT_WORKER_CAPACITY: usize = 1024;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Task handle for tracking asynchronous background execution.
#[must_use = "TaskHandle must be awaited or polled to observe task completion and handle errors"]
pub struct TaskHandle<T = ()> {
    rx: Receiver<Result<T>>,
    done: Arc<AtomicBool>,
}

impl<T> TaskHandle<T> {
    #[inline]
    pub fn new(rx: Receiver<Result<T>>, done: Arc<AtomicBool>) -> Self {
        Self { rx, done }
    }

    /// Checks non-blockingly if the task has completed.
    #[inline]
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Waits for task completion up to the specified timeout (or blocks indefinitely if None).
    pub fn wait(&self, timeout: Option<Duration>) -> Result<T> {
        let res = match timeout {
            Some(t) => self.rx.recv_timeout(t).map_err(|e| match e {
                RecvTimeoutError::Timeout => ScalixError::Timeout,
                RecvTimeoutError::Disconnected => ScalixError::ChannelClosed,
            }),
            None => self
                .rx
                .recv()
                .map_err(|_| ScalixError::ChannelClosed),
        }?;
        res
    }
}

/// In-process worker thread pool for offloading execution and isolating backend contexts.
pub struct WorkerPool {
    sender: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// Creates a new worker pool with the specified number of threads and default PID-scoped prefix (`<pid>/scx-w`).
    pub fn new(num_threads: usize) -> Self {
        let pid = std::process::id();
        Self::with_prefix(&format!("{}/scx-w", pid), num_threads)
    }

    /// Fallible constructor initializing a worker pool with a thread prefix, thread count, and channel capacity.
    pub fn try_with_prefix(name_prefix: &str, num_threads: usize, capacity: usize) -> Result<Self> {
        let (sender, receiver) = bounded::<Job>(capacity);
        let mut workers = Vec::with_capacity(num_threads);

        for id in 0..num_threads {
            let rx = receiver.clone();
            let thread_name = if num_threads == 1 {
                name_prefix.to_string()
            } else {
                format!("{}{}", name_prefix, id)
            };
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                })
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to spawn worker thread: {e}")))?;

            workers.push(handle);
        }

        Ok(Self {
            sender: Some(sender),
            workers,
        })
    }

    /// Creates a new worker pool with custom thread name prefix.
    /// If `num_threads == 1`, the name is `<name_prefix>`.
    /// If `num_threads > 1`, threads are named `<name_prefix><id>`.
    pub fn with_prefix(name_prefix: &str, num_threads: usize) -> Self {
        Self::try_with_prefix(name_prefix, num_threads, DEFAULT_WORKER_CAPACITY)
            .expect("Failed to spawn scalix worker thread")
    }

    /// Submits a closure job to the worker pool.
    pub fn submit<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let sender = self.sender.as_ref().ok_or(ScalixError::ChannelClosed)?;
        sender
            .send(Box::new(f))
            .map_err(|_| ScalixError::ChannelClosed)
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Drop the sender to signal workers to terminate
        drop(self.sender.take());
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}
