//! Zero-Copy DMA-BUF Subsystem
//!
//! Provides platform-gated, hardware-backed DMA buffer allocation and memory mapping
//! for zero-copy data pipelines across CPU, Vulkan, OpenGL/EGL, and V4L2 accelerators.

#[cfg(target_os = "linux")]
pub mod linux_dma_heap;

#[cfg(target_os = "linux")]
pub mod linux_drm;

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
pub mod android_ahb;

use std::os::fd::RawFd;
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

use crate::types::{ImageDesc, ImageDescMut, ImageDimensions, PixelFormat, Result, ScalixError};

/// DMA Buffer cache synchronization flags for `DMA_BUF_IOCTL_SYNC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaSyncFlags {
    Read,
    Write,
    ReadWrite,
}

// Linux DMA-BUF ioctl sync structure and flags
#[cfg(target_os = "linux")]
#[repr(C)]
struct DmaBufSync {
    flags: u64,
}

#[cfg(target_os = "linux")]
const DMA_BUF_SYNC_READ: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const DMA_BUF_SYNC_WRITE: u64 = 2 << 0;
#[cfg(target_os = "linux")]
const DMA_BUF_SYNC_RW: u64 = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE;
#[cfg(target_os = "linux")]
const DMA_BUF_SYNC_START: u64 = 0 << 2;
#[cfg(target_os = "linux")]
const DMA_BUF_SYNC_END: u64 = 1 << 2;
#[cfg(target_os = "linux")]
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x40086200;

/// Unified hardware DMA allocation descriptor.
pub struct DmaAllocation {
    #[cfg(target_os = "linux")]
    pub fd: OwnedFd,

    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    pub ahb_handle: std::ptr::NonNull<std::ffi::c_void>,

    pub host_ptr: std::ptr::NonNull<u8>,
    pub size: usize,
    pub stride: usize,
}

unsafe impl Send for DmaAllocation {}
unsafe impl Sync for DmaAllocation {}

/// Pluggable hardware DMA memory allocator trait.
pub trait DmaAllocator: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn allocate(&self, dimensions: ImageDimensions, format: PixelFormat) -> Result<DmaAllocation>;
}

#[cfg(target_os = "linux")]
fn mmap_fd(fd: &OwnedFd, size: usize) -> Result<NonNull<u8>> {
    let host_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };

    if host_ptr == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error();
        return Err(ScalixError::DmaMapFailed(format!("mmap failed: {errno}")));
    }

    NonNull::new(host_ptr as *mut u8)
        .ok_or_else(|| ScalixError::DmaMapFailed("mmap returned null pointer".to_string()))
}

#[cfg(target_os = "linux")]
impl DmaAllocator for linux_dma_heap::LinuxDmaHeapAllocator {
    #[inline]
    fn name(&self) -> &'static str {
        "linux_dma_heap"
    }

    #[inline]
    fn is_available(&self) -> bool {
        linux_dma_heap::LinuxDmaHeapAllocator::is_available()
    }

    fn allocate(&self, dimensions: ImageDimensions, format: PixelFormat) -> Result<DmaAllocation> {
        let (fd, size, stride) = Self::allocate_dimensions(dimensions, format)?;
        let host_ptr = mmap_fd(&fd, size)?;
        Ok(DmaAllocation {
            fd,
            host_ptr,
            size,
            stride,
        })
    }
}

#[cfg(target_os = "linux")]
impl DmaAllocator for linux_drm::LinuxDrmAllocator {
    #[inline]
    fn name(&self) -> &'static str {
        "linux_drm"
    }

    #[inline]
    fn is_available(&self) -> bool {
        linux_drm::LinuxDrmAllocator::is_available()
    }

    fn allocate(&self, dimensions: ImageDimensions, format: PixelFormat) -> Result<DmaAllocation> {
        let (fd, size, stride) = Self::allocate_dimensions(dimensions, format)?;
        let host_ptr = mmap_fd(&fd, size)?;
        Ok(DmaAllocation {
            fd,
            host_ptr,
            size,
            stride,
        })
    }
}

#[cfg(all(target_os = "android", target_arch = "aarch64"))]
impl DmaAllocator for android_ahb::AndroidAhbAllocator {
    #[inline]
    fn name(&self) -> &'static str {
        "android_ahb"
    }

    #[inline]
    fn is_available(&self) -> bool {
        android_ahb::AndroidAhbAllocator::is_available()
    }

    fn allocate(&self, dimensions: ImageDimensions, format: PixelFormat) -> Result<DmaAllocation> {
        let (ahb_ptr, host_ptr, size, stride) =
            Self::allocate_dimensions(dimensions, format)?;
        let ahb_handle = std::ptr::NonNull::new(ahb_ptr)
            .ok_or_else(|| ScalixError::DmaAllocationFailed("Null AHardwareBuffer pointer".to_string()))?;
        let host_ptr = std::ptr::NonNull::new(host_ptr)
            .ok_or_else(|| ScalixError::DmaMapFailed("Null virtual address returned from AHardwareBuffer_lock".to_string()))?;
        Ok(DmaAllocation {
            ahb_handle,
            host_ptr,
            size,
            stride,
        })
    }
}

/// Represents an allocated, memory-mapped hardware DMA buffer.
pub struct DmaBuffer {
    #[cfg(target_os = "linux")]
    fd: OwnedFd,

    #[cfg(all(target_os = "android", target_arch = "aarch64"))]
    ahb_handle: *mut std::ffi::c_void,

    host_ptr: *mut u8,
    size: usize,
    dimensions: ImageDimensions,
    stride: usize,
    format: PixelFormat,
}

// DmaBuffer owns the memory mapping and underlying kernel fd/handle
unsafe impl Send for DmaBuffer {}
unsafe impl Sync for DmaBuffer {}

impl DmaBuffer {
    /// Constructs a `DmaBuffer` taking ownership of an existing hardware `DmaAllocation`.
    #[inline]
    #[must_use]
    pub fn from_allocation(
        alloc: DmaAllocation,
        dimensions: ImageDimensions,
        format: PixelFormat,
    ) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            fd: alloc.fd,
            #[cfg(all(target_os = "android", target_arch = "aarch64"))]
            ahb_handle: alloc.ahb_handle.as_ptr(),
            host_ptr: alloc.host_ptr.as_ptr(),
            size: alloc.size,
            dimensions,
            stride: alloc.stride,
            format,
        }
    }

    /// Allocates a new hardware DMA buffer for the given spatial dimensions and pixel format.
    ///
    /// On Linux: Probes `/dev/dma_heap/*` (DMA-Heap) first, then falls back to `/dev/dri/renderD128` (DRM Dumb).
    /// On Android: Uses `AHardwareBuffer` (API 26+, aarch64 only).
    pub fn allocate_dimensions(dimensions: ImageDimensions, format: PixelFormat) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(ScalixError::InvalidDimensions {
                width: dimensions.width,
                height: dimensions.height,
            });
        }

        #[cfg(target_os = "linux")]
        {
            let heap_allocator = linux_dma_heap::LinuxDmaHeapAllocator;
            match heap_allocator.allocate(dimensions, format) {
                Ok(alloc) => Ok(Self::from_allocation(alloc, dimensions, format)),
                Err(heap_err) => {
                    log::debug!("DMA-Heap allocation skipped or unavailable: {heap_err}, trying DRM...");
                    let drm_allocator = linux_drm::LinuxDrmAllocator;
                    drm_allocator
                        .allocate(dimensions, format)
                        .map(|alloc| Self::from_allocation(alloc, dimensions, format))
                        .map_err(|drm_err| {
                            ScalixError::DmaUnavailable(format!(
                                "All Linux DMA allocators failed. DMA-Heap: [{heap_err}], DRM: [{drm_err}]"
                            ))
                        })
                }
            }
        }

        #[cfg(all(target_os = "android", target_arch = "aarch64"))]
        {
            let ahb_allocator = android_ahb::AndroidAhbAllocator;
            let alloc = ahb_allocator.allocate(dimensions, format)?;
            Ok(Self::from_allocation(alloc, dimensions, format))
        }

        #[cfg(not(any(target_os = "linux", all(target_os = "android", target_arch = "aarch64"))))]
        {
            let _ = (dimensions, format);
            Err(ScalixError::DmaUnavailable(
                "DMA buffer allocation is only supported on Linux (kernel 5.6+) and Android (aarch64)".to_string(),
            ))
        }
    }

    /// Allocates a new hardware DMA buffer for scalar width and height.
    #[inline]
    pub fn allocate(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        Self::allocate_dimensions(ImageDimensions::new(width, height), format)
    }

    /// Returns the kernel DMA-BUF file descriptor on Linux, or -1 on platforms without an fd.
    #[inline]
    #[must_use]
    pub fn fd(&self) -> RawFd {
        #[cfg(target_os = "linux")]
        {
            self.fd.as_raw_fd()
        }
        #[cfg(not(target_os = "linux"))]
        {
            -1
        }
    }

    /// Returns the raw user-space mapped memory pointer.
    #[inline]
    #[must_use]
    pub const fn host_ptr(&self) -> *mut u8 {
        self.host_ptr
    }

    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.dimensions.width
    }

    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.dimensions.height
    }

    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns the CPU memory slice for reading.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.size) }
    }

    /// Returns the CPU memory slice for writing.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.size) }
    }

    /// Signals the start of CPU access for cache coherency.
    pub fn sync_start(&self, flags: DmaSyncFlags) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let sync_flags = match flags {
                DmaSyncFlags::Read => DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START,
                DmaSyncFlags::Write => DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_START,
                DmaSyncFlags::ReadWrite => DMA_BUF_SYNC_RW | DMA_BUF_SYNC_START,
            };
            let mut sync = DmaBufSync { flags: sync_flags };
            let ret = unsafe {
                libc::ioctl(
                    self.fd.as_raw_fd(),
                    DMA_BUF_IOCTL_SYNC,
                    &mut sync as *mut DmaBufSync,
                )
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(ScalixError::DmaSyncFailed(format!("DMA_BUF_SYNC_START failed: {err}")));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = flags;
            Ok(())
        }
    }

    /// Signals the completion of CPU access to flush/invalidate caches for hardware accelerator visibility.
    pub fn sync_end(&self, flags: DmaSyncFlags) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let sync_flags = match flags {
                DmaSyncFlags::Read => DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END,
                DmaSyncFlags::Write => DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_END,
                DmaSyncFlags::ReadWrite => DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END,
            };
            let mut sync = DmaBufSync { flags: sync_flags };
            let ret = unsafe {
                libc::ioctl(
                    self.fd.as_raw_fd(),
                    DMA_BUF_IOCTL_SYNC,
                    &mut sync as *mut DmaBufSync,
                )
            };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(ScalixError::DmaSyncFailed(format!("DMA_BUF_SYNC_END failed: {err}")));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = flags;
            Ok(())
        }
    }

    /// Executes a closure with synchronized CPU write access, automatically managing `sync_start` and `sync_end`.
    pub fn with_write<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        self.sync_start(DmaSyncFlags::Write)?;
        let slice = unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.size) };
        let result = f(slice);
        self.sync_end(DmaSyncFlags::Write)?;
        Ok(result)
    }

    /// Executes a closure with synchronized CPU read access, automatically managing `sync_start` and `sync_end`.
    pub fn with_read<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R,
    {
        self.sync_start(DmaSyncFlags::Read)?;
        let slice = unsafe { std::slice::from_raw_parts(self.host_ptr, self.size) };
        let result = f(slice);
        self.sync_end(DmaSyncFlags::Read)?;
        Ok(result)
    }

    /// Creates an immutable `ImageDesc` view referencing this DMA buffer.
    #[inline]
    #[must_use]
    pub fn as_image_desc(&self) -> ImageDesc<'_> {
        let fd_raw = self.fd();
        let mut desc = ImageDesc::new(
            self.dimensions.width,
            self.dimensions.height,
            self.stride,
            self.format,
            self.as_slice(),
        )
        .expect("DmaBuffer dimensions and stride are guaranteed valid");

        if fd_raw >= 0 {
            desc = desc.with_dma_buf(fd_raw);
        }
        desc
    }

    /// Creates a mutable `ImageDescMut` view referencing this DMA buffer.
    #[inline]
    #[must_use]
    pub fn as_image_desc_mut(&mut self) -> ImageDescMut<'_> {
        let fd_raw = self.fd();
        let width = self.dimensions.width;
        let height = self.dimensions.height;
        let stride = self.stride;
        let format = self.format;
        let mut desc = ImageDescMut::new(
            width,
            height,
            stride,
            format,
            self.as_mut_slice(),
        )
        .expect("DmaBuffer dimensions and stride are guaranteed valid");

        if fd_raw >= 0 {
            desc = desc.with_dma_buf(fd_raw);
        }
        desc
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if !self.host_ptr.is_null() && self.host_ptr != libc::MAP_FAILED as *mut u8 {
                unsafe {
                    libc::munmap(self.host_ptr as *mut libc::c_void, self.size);
                }
                self.host_ptr = std::ptr::null_mut();
            }
            // self.fd is closed automatically by OwnedFd drop
        }

        #[cfg(all(target_os = "android", target_arch = "aarch64"))]
        {
            if !self.ahb_handle.is_null() {
                unsafe {
                    if !self.host_ptr.is_null() {
                        let mut fence: i32 = -1;
                        let _ = android_ahb::AHardwareBuffer_unlock(self.ahb_handle, &mut fence);
                        self.host_ptr = std::ptr::null_mut();
                    }
                    android_ahb::AHardwareBuffer_release(self.ahb_handle);
                    self.ahb_handle = std::ptr::null_mut();
                }
            }
        }
    }
}
