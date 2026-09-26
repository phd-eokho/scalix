//! Zero-Copy DMA-BUF Subsystem
//!
//! Provides platform-gated, hardware-backed DMA buffer allocation and memory mapping
//! for zero-copy data pipelines across CPU, Vulkan, OpenGL/EGL, and V4L2 accelerators.

#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod linux_dma_heap;

#[cfg(target_os = "linux")]
pub mod linux_drm;

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub mod android_ahb;

use std::os::fd::RawFd;
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(any(target_os = "linux", target_os = "android"))]
use std::ptr::NonNull;

use crate::types::{ImageDesc, ImageDescMut, ImageDimensions, PixelFormat, Result, ScalixError};

/// DMA Buffer cache synchronization flags for `DMA_BUF_IOCTL_SYNC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaSyncFlags {
    Read,
    Write,
    ReadWrite,
}

// Linux/Android DMA-BUF ioctl sync structure and flags
#[cfg(any(target_os = "linux", target_os = "android"))]
#[repr(C)]
struct DmaBufSync {
    flags: u64,
}

#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_SYNC_READ: u64 = 1 << 0;
#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_SYNC_WRITE: u64 = 1 << 1;
#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_SYNC_RW: u64 = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE;
#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_SYNC_START: u64 = 0 << 2;
#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_SYNC_END: u64 = 1 << 2;
#[cfg(any(target_os = "linux", target_os = "android"))]
const DMA_BUF_IOCTL_SYNC: libc::c_ulong = 0x40086200;

/// Unified hardware DMA allocation descriptor.
pub struct DmaAllocation {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fd: Option<OwnedFd>,

    #[cfg(all(
        target_os = "android",
        any(target_arch = "aarch64", target_arch = "arm")
    ))]
    pub ahb_handle: Option<std::ptr::NonNull<std::ffi::c_void>>,

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

#[cfg(any(target_os = "linux", target_os = "android"))]
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

    if host_ptr == libc::MAP_FAILED || host_ptr.is_null() {
        let errno = std::io::Error::last_os_error();
        return Err(ScalixError::DmaMapFailed(format!("mmap failed: {errno}")));
    }

    NonNull::new(host_ptr as *mut u8)
        .ok_or_else(|| ScalixError::DmaMapFailed("mmap returned null pointer".to_string()))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
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
            fd: Some(fd),
            #[cfg(all(
                target_os = "android",
                any(target_arch = "aarch64", target_arch = "arm")
            ))]
            ahb_handle: None,
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
            fd: Some(fd),
            host_ptr,
            size,
            stride,
        })
    }
}

#[cfg(all(
    target_os = "android",
    any(target_arch = "aarch64", target_arch = "arm")
))]
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
        let (ahb_ptr, host_ptr, size, stride) = Self::allocate_dimensions(dimensions, format)?;
        let ahb_handle = std::ptr::NonNull::new(ahb_ptr).ok_or_else(|| {
            ScalixError::DmaAllocationFailed("Null AHardwareBuffer pointer".to_string())
        })?;
        let host_ptr = std::ptr::NonNull::new(host_ptr).ok_or_else(|| {
            ScalixError::DmaMapFailed(
                "Null virtual address returned from AHardwareBuffer_lock".to_string(),
            )
        })?;
        Ok(DmaAllocation {
            fd: None,
            ahb_handle: Some(ahb_handle),
            host_ptr,
            size,
            stride,
        })
    }
}

/// Selection type for hardware DMA buffer memory allocators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub enum DmaAllocatorType {
    #[default]
    Auto = 0,
    DmaHeap = 1,
    DrmDumb = 2,
    AndroidAhb = 3,
    HostAligned = 4,
}

/// Represents an allocated, memory-mapped hardware DMA buffer.
pub struct DmaBuffer {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fd: Option<OwnedFd>,
    imported_fd: RawFd,

    #[cfg(all(
        target_os = "android",
        any(target_arch = "aarch64", target_arch = "arm")
    ))]
    ahb_handle: *mut std::ffi::c_void,

    host_ptr: *mut u8,
    size: usize,
    dimensions: ImageDimensions,
    stride: usize,
    format: PixelFormat,
    allocator_type: DmaAllocatorType,
    is_host_aligned: bool,
    is_imported: bool,
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
        allocator_type: DmaAllocatorType,
    ) -> Self {
        Self {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            fd: alloc.fd,
            imported_fd: -1,
            #[cfg(all(
                target_os = "android",
                any(target_arch = "aarch64", target_arch = "arm")
            ))]
            ahb_handle: alloc
                .ahb_handle
                .map(|h| h.as_ptr())
                .unwrap_or(std::ptr::null_mut()),
            host_ptr: alloc.host_ptr.as_ptr(),
            size: alloc.size,
            dimensions,
            stride: alloc.stride,
            format,
            allocator_type,
            is_host_aligned: false,
            is_imported: false,
        }
    }

    /// Wraps an externally provided raw Linux / Android DMA-BUF file descriptor with layout metadata.
    ///
    /// Maps the DMA-BUF memory for CPU host access and manages DMA-BUF cache sync ioctls.
    /// Does not take ownership of closing `fd` on drop.
    pub fn from_raw_dma_buf(
        fd: RawFd,
        dimensions: ImageDimensions,
        format: PixelFormat,
        stride_bytes: Option<usize>,
    ) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(ScalixError::InvalidDimensions {
                width: dimensions.width,
                height: dimensions.height,
            });
        }
        if fd < 0 {
            return Err(ScalixError::DmaAllocationFailed(
                "Invalid negative DMA-BUF file descriptor".to_string(),
            ));
        }

        let stride = match stride_bytes {
            Some(s) => {
                let min = format.min_stride(dimensions.width)?;
                if s < min {
                    return Err(ScalixError::InvalidStride {
                        stride: s,
                        min_stride: min,
                    });
                }
                s
            }
            None => format.min_stride(dimensions.width)?,
        };
        let size = format.min_buffer_size_dims(dimensions, stride)?;

        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let host_ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    0,
                )
            };

            if host_ptr == libc::MAP_FAILED || host_ptr.is_null() {
                let errno = std::io::Error::last_os_error();
                return Err(ScalixError::DmaMapFailed(format!(
                    "mmap of external DMA-BUF fd {fd} failed: {errno}"
                )));
            }

            Ok(Self {
                fd: None,
                imported_fd: fd,
                #[cfg(all(
                    target_os = "android",
                    any(target_arch = "aarch64", target_arch = "arm")
                ))]
                ahb_handle: std::ptr::null_mut(),
                host_ptr: host_ptr as *mut u8,
                size,
                dimensions,
                stride,
                format,
                allocator_type: DmaAllocatorType::DmaHeap,
                is_host_aligned: false,
                is_imported: true,
            })
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            Err(ScalixError::DmaUnavailable(
                "DMA-BUF import is only supported on Linux and Android".to_string(),
            ))
        }
    }

    /// Allocates a new hardware DMA buffer with a specific allocator selection.
    pub fn allocate_dimensions_with_type(
        dimensions: ImageDimensions,
        format: PixelFormat,
        alloc_type: DmaAllocatorType,
    ) -> Result<Self> {
        if dimensions.is_empty() {
            return Err(ScalixError::InvalidDimensions {
                width: dimensions.width,
                height: dimensions.height,
            });
        }

        match alloc_type {
            DmaAllocatorType::Auto => {
                #[cfg(target_os = "linux")]
                {
                    let heap_allocator = linux_dma_heap::LinuxDmaHeapAllocator;
                    match heap_allocator.allocate(dimensions, format) {
                        Ok(alloc) => Ok(Self::from_allocation(
                            alloc,
                            dimensions,
                            format,
                            DmaAllocatorType::DmaHeap,
                        )),
                        Err(heap_err) => {
                            log::debug!(
                                "DMA-Heap allocation skipped: {heap_err}, trying DRM Dumb..."
                            );
                            let drm_allocator = linux_drm::LinuxDrmAllocator;
                            match drm_allocator.allocate(dimensions, format) {
                                Ok(alloc) => Ok(Self::from_allocation(
                                    alloc,
                                    dimensions,
                                    format,
                                    DmaAllocatorType::DrmDumb,
                                )),
                                Err(drm_err) => {
                                    Err(ScalixError::DmaUnavailable(format!(
                                        "All Linux DMA allocators failed. DMA-Heap: [{heap_err}], DRM: [{drm_err}]"
                                    )))
                                }
                            }
                        }
                    }
                }
                #[cfg(all(
                    target_os = "android",
                    any(target_arch = "aarch64", target_arch = "arm")
                ))]
                {
                    // 1. Try AHardwareBuffer (NDK mode)
                    let ahb_allocator = android_ahb::AndroidAhbAllocator;
                    match ahb_allocator.allocate(dimensions, format) {
                        Ok(alloc) => Ok(Self::from_allocation(
                            alloc,
                            dimensions,
                            format,
                            DmaAllocatorType::AndroidAhb,
                        )),
                        Err(ahb_err) => {
                            log::debug!("AHB allocation failed: {ahb_err}, probing DMA-Heap for Vendor mode...");
                            // 2. Try DMA-Heap (/dev/dma_heap/*) for pure Vendor mode
                            let heap_allocator = linux_dma_heap::LinuxDmaHeapAllocator;
                            match heap_allocator.allocate(dimensions, format) {
                                Ok(alloc) => Ok(Self::from_allocation(
                                    alloc,
                                    dimensions,
                                    format,
                                    DmaAllocatorType::DmaHeap,
                                )),
                                Err(heap_err) => {
                                    Err(ScalixError::DmaUnavailable(format!(
                                        "All Android DMA allocators failed. AHB: [{ahb_err}], DMA-Heap: [{heap_err}]"
                                    )))
                                }
                            }
                        }
                    }
                }
                #[cfg(not(any(
                    target_os = "linux",
                    all(
                        target_os = "android",
                        any(target_arch = "aarch64", target_arch = "arm")
                    )
                )))]
                {
                    Err(ScalixError::DmaUnavailable(
                        "Hardware DMA buffer allocation is only supported on Linux (kernel 5.6+) and Android (API 26+)".to_string(),
                    ))
                }
            }
            DmaAllocatorType::DmaHeap => {
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    let heap_allocator = linux_dma_heap::LinuxDmaHeapAllocator;
                    let alloc = heap_allocator.allocate(dimensions, format)?;
                    Ok(Self::from_allocation(
                        alloc,
                        dimensions,
                        format,
                        DmaAllocatorType::DmaHeap,
                    ))
                }
                #[cfg(not(any(target_os = "linux", target_os = "android")))]
                {
                    Err(ScalixError::DmaUnavailable(
                        "DMA-Heap allocator is only available on Linux and Android".to_string(),
                    ))
                }
            }
            DmaAllocatorType::DrmDumb => {
                #[cfg(target_os = "linux")]
                {
                    let drm_allocator = linux_drm::LinuxDrmAllocator;
                    let alloc = drm_allocator.allocate(dimensions, format)?;
                    Ok(Self::from_allocation(
                        alloc,
                        dimensions,
                        format,
                        DmaAllocatorType::DrmDumb,
                    ))
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(ScalixError::DmaUnavailable(
                        "DRM Dumb allocator is only available on Linux".to_string(),
                    ))
                }
            }
            DmaAllocatorType::AndroidAhb => {
                #[cfg(all(
                    target_os = "android",
                    any(target_arch = "aarch64", target_arch = "arm")
                ))]
                {
                    let ahb_allocator = android_ahb::AndroidAhbAllocator;
                    let alloc = ahb_allocator.allocate(dimensions, format)?;
                    Ok(Self::from_allocation(
                        alloc,
                        dimensions,
                        format,
                        DmaAllocatorType::AndroidAhb,
                    ))
                }
                #[cfg(not(all(
                    target_os = "android",
                    any(target_arch = "aarch64", target_arch = "arm")
                )))]
                {
                    Err(ScalixError::DmaUnavailable(
                        "Android AHardwareBuffer allocator is only available on Android"
                            .to_string(),
                    ))
                }
            }
            DmaAllocatorType::HostAligned => {
                let stride = format.min_stride(dimensions.width)?;
                let size = format.min_buffer_size_dims(dimensions, stride)?;
                let mut aligned_ptr: *mut libc::c_void = std::ptr::null_mut();
                let ret = unsafe {
                    libc::posix_memalign(
                        &mut aligned_ptr,
                        crate::types::REQUIRED_MEMORY_ALIGNMENT,
                        size,
                    )
                };
                if ret != 0 || aligned_ptr.is_null() {
                    return Err(ScalixError::DmaAllocationFailed(
                        "Failed to allocate 64-byte aligned host memory".to_string(),
                    ));
                }
                unsafe {
                    std::ptr::write_bytes(aligned_ptr as *mut u8, 0, size);
                }
                Ok(Self {
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    fd: None,
                    imported_fd: -1,
                    #[cfg(all(
                        target_os = "android",
                        any(target_arch = "aarch64", target_arch = "arm")
                    ))]
                    ahb_handle: std::ptr::null_mut(),
                    host_ptr: aligned_ptr as *mut u8,
                    size,
                    dimensions,
                    stride,
                    format,
                    allocator_type: DmaAllocatorType::HostAligned,
                    is_host_aligned: true,
                    is_imported: false,
                })
            }
        }
    }

    /// Allocates a new hardware DMA buffer for the given spatial dimensions and pixel format using Auto discovery.
    #[inline]
    pub fn allocate_dimensions(dimensions: ImageDimensions, format: PixelFormat) -> Result<Self> {
        Self::allocate_dimensions_with_type(dimensions, format, DmaAllocatorType::Auto)
    }

    /// Allocates a new hardware DMA buffer for scalar width and height using specific allocator type.
    #[inline]
    pub fn allocate_with_type(
        width: u32,
        height: u32,
        format: PixelFormat,
        alloc_type: DmaAllocatorType,
    ) -> Result<Self> {
        Self::allocate_dimensions_with_type(ImageDimensions::new(width, height), format, alloc_type)
    }

    /// Allocates a new hardware DMA buffer for scalar width and height using Auto discovery.
    #[inline]
    pub fn allocate(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        Self::allocate_dimensions(ImageDimensions::new(width, height), format)
    }

    /// Returns the allocator type used to back this buffer.
    #[inline]
    #[must_use]
    pub const fn allocator_type(&self) -> DmaAllocatorType {
        self.allocator_type
    }

    /// Returns true if this buffer was imported from an external DMA-BUF file descriptor.
    #[inline]
    #[must_use]
    pub const fn is_imported(&self) -> bool {
        self.is_imported
    }

    /// Returns the kernel DMA-BUF file descriptor on Linux/Android, or -1 on platforms without an fd.
    #[inline]
    #[must_use]
    pub fn fd(&self) -> RawFd {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if let Some(ref f) = self.fd {
                f.as_raw_fd()
            } else {
                self.imported_fd
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            -1
        }
    }

    /// Returns the raw Android AHardwareBuffer handle on Android, or null on other platforms.
    #[inline]
    #[must_use]
    pub fn ahb_handle(&self) -> *mut std::ffi::c_void {
        #[cfg(all(
            target_os = "android",
            any(target_arch = "aarch64", target_arch = "arm")
        ))]
        {
            self.ahb_handle
        }
        #[cfg(not(all(
            target_os = "android",
            any(target_arch = "aarch64", target_arch = "arm")
        )))]
        {
            std::ptr::null_mut()
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
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let raw_fd = self.fd();
            if raw_fd >= 0 {
                let sync_flags = match flags {
                    DmaSyncFlags::Read => DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START,
                    DmaSyncFlags::Write => DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_START,
                    DmaSyncFlags::ReadWrite => DMA_BUF_SYNC_RW | DMA_BUF_SYNC_START,
                };
                let mut sync = DmaBufSync { flags: sync_flags };
                // NOTE: Cast to `as _` because libc::ioctl `request` is c_int on Android and c_ulong on Linux
                let ret = unsafe {
                    libc::ioctl(
                        raw_fd,
                        DMA_BUF_IOCTL_SYNC as _,
                        &mut sync as *mut DmaBufSync,
                    )
                };
                if ret != 0 {
                    let err = std::io::Error::last_os_error();
                    return Err(ScalixError::DmaSyncFailed(format!(
                        "DMA_BUF_SYNC_START failed: {err}"
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = flags;
            Ok(())
        }
    }

    /// Signals the completion of CPU access to flush/invalidate caches for hardware accelerator visibility.
    pub fn sync_end(&self, flags: DmaSyncFlags) -> Result<()> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let raw_fd = self.fd();
            if raw_fd >= 0 {
                let sync_flags = match flags {
                    DmaSyncFlags::Read => DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END,
                    DmaSyncFlags::Write => DMA_BUF_SYNC_WRITE | DMA_BUF_SYNC_END,
                    DmaSyncFlags::ReadWrite => DMA_BUF_SYNC_RW | DMA_BUF_SYNC_END,
                };
                let mut sync = DmaBufSync { flags: sync_flags };
                // NOTE: Cast to `as _` because libc::ioctl `request` is c_int on Android and c_ulong on Linux
                let ret = unsafe {
                    libc::ioctl(
                        raw_fd,
                        DMA_BUF_IOCTL_SYNC as _,
                        &mut sync as *mut DmaBufSync,
                    )
                };
                if ret != 0 {
                    let err = std::io::Error::last_os_error();
                    return Err(ScalixError::DmaSyncFailed(format!(
                        "DMA_BUF_SYNC_END failed: {err}"
                    )));
                }
            }
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
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
        let mut desc = ImageDescMut::new(width, height, stride, format, self.as_mut_slice())
            .expect("DmaBuffer dimensions and stride are guaranteed valid");

        if fd_raw >= 0 {
            desc = desc.with_dma_buf(fd_raw);
        }
        desc
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        if self.is_host_aligned {
            if !self.host_ptr.is_null() {
                unsafe {
                    libc::free(self.host_ptr as *mut libc::c_void);
                }
                self.host_ptr = std::ptr::null_mut();
            }
            return;
        }

        #[cfg(all(
            target_os = "android",
            any(target_arch = "aarch64", target_arch = "arm")
        ))]
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
                return;
            }
        }

        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if !self.host_ptr.is_null() && self.host_ptr != libc::MAP_FAILED as *mut u8 {
                unsafe {
                    libc::munmap(self.host_ptr as *mut libc::c_void, self.size);
                }
                self.host_ptr = std::ptr::null_mut();
            }
            // self.fd (if any) is closed automatically when OwnedFd drops
        }
    }
}
