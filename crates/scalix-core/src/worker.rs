use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::types::{Result, ScalixError};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Task handle for tracking asynchronous background execution.
pub struct TaskHandle<T = ()> {
    rx: Receiver<Result<T>>,
    done: Arc<AtomicBool>,
}

impl<T> TaskHandle<T> {
    pub fn new(rx: Receiver<Result<T>>, done: Arc<AtomicBool>) -> Self {
        Self { rx, done }
    }

    /// Checks non-blockingly if the task has completed.
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Waits for task completion up to the specified timeout (or blocks indefinitely if None).
    pub fn wait(self, timeout: Option<Duration>) -> Result<T> {
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

    /// Creates a new worker pool with custom thread name prefix.
    /// If `num_threads == 1`, the name is `<name_prefix>`.
    /// If `num_threads > 1`, threads are named `<name_prefix><id>`.
    pub fn with_prefix(name_prefix: &str, num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        let mut workers = Vec::with_capacity(num_threads);

        for id in 0..num_threads {
            let rx = Arc::clone(&receiver);
            let thread_name = if num_threads == 1 {
                name_prefix.to_string()
            } else {
                format!("{}{}", name_prefix, id)
            };
            let handle = thread::Builder::new()
                .name(thread_name)
                .spawn(move || loop {
                    let job = {
                        let lock = rx.lock().unwrap();
                        lock.recv()
                    };

                    match job {
                        Ok(job) => job(),
                        Err(_) => break, // Channel disconnected, terminate thread
                    }
                })
                .expect("Failed to spawn scalix worker thread");

            workers.push(handle);
        }

        Self {
            sender: Some(sender),
            workers,
        }
    }

    /// Submits a closure job to the worker pool.
    pub fn submit<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(ref sender) = self.sender {
            sender
                .send(Box::new(f))
                .map_err(|_| ScalixError::ChannelClosed)
        } else {
            Err(ScalixError::ChannelClosed)
        }
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
