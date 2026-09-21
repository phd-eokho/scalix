//! Linux DMA-Heap Allocator (`/dev/dma_heap/system`, `/dev/dma_heap/cma`, Linux 5.6+)

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use crate::types::{ImageDimensions, PixelFormat, Result, ScalixError};

#[repr(C)]
struct DmaHeapAllocationData {
    len: u64,
    fd: u32,
    fd_flags: u32,
    heap_flags: u64,
}

// _IOWR('H', 0x0, struct DmaHeapAllocationData) -> 0xc0184800
const DMA_HEAP_IOCTL_ALLOC: libc::c_ulong = 0xc0184800;

pub(crate) const DMA_HEAP_CANDIDATE_PATHS: &[&str] = &[
    "/dev/dma_heap/system",
    "/dev/dma_heap/system-uncached",
    "/dev/dma_heap/cma",
    "/dev/dma_heap/linux,cma",
    "/dev/dma_heap/reserved",
];

pub struct LinuxDmaHeapAllocator;

impl LinuxDmaHeapAllocator {
    /// Checks whether any DMA-Heap device node is currently accessible.
    #[inline]
    #[must_use]
    pub fn is_available() -> bool {
        DMA_HEAP_CANDIDATE_PATHS
            .iter()
            .any(|path| OpenOptions::new().read(true).write(true).open(path).is_ok())
    }

    /// Attempts to probe and allocate a DMA-BUF file descriptor from available DMA-Heap device nodes.
    #[inline]
    pub fn allocate(
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<(OwnedFd, usize, usize)> {
        Self::allocate_dimensions(ImageDimensions::new(width, height), format)
    }

    /// Attempts to probe and allocate a DMA-BUF file descriptor for given image dimensions.
    pub fn allocate_dimensions(
        dimensions: ImageDimensions,
        format: PixelFormat,
    ) -> Result<(OwnedFd, usize, usize)> {
        let stride = format.min_stride(dimensions.width)?;
        let size = format.min_buffer_size_dims(dimensions, stride)?;

        // Find the first accessible DMA-Heap device node
        let mut last_err = String::from("No DMA-Heap device nodes accessible");
        for &path in DMA_HEAP_CANDIDATE_PATHS {
            let file = match OpenOptions::new().read(true).write(true).open(path) {
                Ok(f) => f,
                Err(e) => {
                    last_err = format!("{}: {}", path, e);
                    continue;
                }
            };

            let mut alloc_data = DmaHeapAllocationData {
                len: size as u64,
                fd: 0,
                fd_flags: libc::O_CLOEXEC as u32 | libc::O_RDWR as u32,
                heap_flags: 0,
            };

            let ret = unsafe {
                libc::ioctl(
                    file.as_raw_fd(),
                    DMA_HEAP_IOCTL_ALLOC,
                    &mut alloc_data as *mut DmaHeapAllocationData,
                )
            };

            if ret == 0 {
                let dma_fd = unsafe { OwnedFd::from_raw_fd(alloc_data.fd as RawFd) };
                return Ok((dma_fd, size, stride));
            } else {
                let errno = std::io::Error::last_os_error();
                last_err = format!("ioctl(DMA_HEAP_IOCTL_ALLOC) on {} failed: {}", path, errno);
            }
        }

        Err(ScalixError::DmaUnavailable(last_err))
    }
}
