use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

impl ImageDimensions {
    #[inline]
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    #[must_use]
    pub const fn checked_area(self) -> Option<usize> {
        (self.width as usize).checked_mul(self.height as usize)
    }

    #[inline]
    pub fn validate_even(self) -> Result<()> {
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            Err(ScalixError::InvalidDimensions {
                width: self.width,
                height: self.height,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ImageGeometry {
    pub dimensions: ImageDimensions,
    pub stride: usize,
    pub format: PixelFormat,
}

impl ImageGeometry {
    #[inline]
    pub fn new(dimensions: ImageDimensions, format: PixelFormat) -> Result<Self> {
        let stride = format.min_stride(dimensions.width)?;
        Ok(Self {
            dimensions,
            stride,
            format,
        })
    }

    #[inline]
    pub fn min_buffer_size(&self) -> Result<usize> {
        self.format
            .min_buffer_size_dims(self.dimensions, self.stride)
    }
}

/// Pixel format definitions supported across backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum PixelFormat {
    Rgba8888 = 0,
    Bgra8888 = 1,
    Rgb888 = 2,
    Bgr888 = 3,
    R8 = 4,
    Rg88 = 5,
    Nv12 = 6,
    Yuv420p = 7,
    Rgba16f = 8,
    Rgba32f = 9,
}

impl PixelFormat {
    /// Returns the bytes per pixel for packed formats.
    /// For planar/semi-planar formats (NV12, YUV420p), returns None.
    #[inline]
    #[must_use]
    pub const fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            PixelFormat::Rgba8888 | PixelFormat::Bgra8888 => Some(4),
            PixelFormat::Rgb888 | PixelFormat::Bgr888 => Some(3),
            PixelFormat::R8 => Some(1),
            PixelFormat::Rg88 => Some(2),
            PixelFormat::Rgba16f => Some(8),
            PixelFormat::Rgba32f => Some(16),
            PixelFormat::Nv12 | PixelFormat::Yuv420p => None,
        }
    }

    /// Computes minimum byte stride for a given width.
    #[inline]
    pub fn min_stride(self, width: u32) -> Result<usize> {
        if width == 0 {
            return Err(ScalixError::InvalidDimensions {
                width: 0,
                height: 0,
            });
        }
        if let Some(bpp) = self.bytes_per_pixel() {
            (width as usize)
                .checked_mul(bpp)
                .ok_or(ScalixError::InvalidDimensions { width, height: 0 })
        } else {
            // For NV12 and YUV420p, Y plane stride is width bytes
            Ok(width as usize)
        }
    }

    /// Computes minimum required buffer size for an image of given dimensions and stride.
    #[inline]
    pub fn min_buffer_size(self, width: u32, height: u32, stride: usize) -> Result<usize> {
        self.min_buffer_size_dims(ImageDimensions::new(width, height), stride)
    }

    /// Computes minimum required buffer size for given image dimensions and stride.
    pub fn min_buffer_size_dims(self, dims: ImageDimensions, stride: usize) -> Result<usize> {
        if dims.is_empty() {
            return Err(ScalixError::InvalidDimensions {
                width: dims.width,
                height: dims.height,
            });
        }
        let min_s = self.min_stride(dims.width)?;
        if stride < min_s {
            return Err(ScalixError::InvalidStride {
                stride,
                min_stride: min_s,
            });
        }

        match self {
            PixelFormat::Nv12 | PixelFormat::Yuv420p => self.min_planar_size(dims, stride),
            _ => self.min_packed_size(dims, stride, min_s),
        }
    }

    #[inline]
    fn min_planar_size(self, dims: ImageDimensions, stride: usize) -> Result<usize> {
        dims.validate_even()?;
        let y_size =
            stride
                .checked_mul(dims.height as usize)
                .ok_or(ScalixError::InvalidDimensions {
                    width: dims.width,
                    height: dims.height,
                })?;
        let uv_size = y_size / 2;
        y_size
            .checked_add(uv_size)
            .ok_or(ScalixError::InvalidDimensions {
                width: dims.width,
                height: dims.height,
            })
    }

    #[inline]
    fn min_packed_size(self, dims: ImageDimensions, stride: usize, min_s: usize) -> Result<usize> {
        let preceding = stride
            .checked_mul((dims.height as usize).saturating_sub(1))
            .ok_or(ScalixError::InvalidDimensions {
                width: dims.width,
                height: dims.height,
            })?;
        preceding
            .checked_add(min_s)
            .ok_or(ScalixError::InvalidDimensions {
                width: dims.width,
                height: dims.height,
            })
    }
}

/// Image scaling filter and interpolation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum FilterMode {
    /// Nearest neighbor sampling (0-order hold)
    Nearest = 0,
    /// Bilinear interpolation (1st-order tensor product)
    Bilinear = 1,
    /// Bicubic interpolation using 2D Catmull-Rom cubic spline ($a = -0.5$) (Keys, 1981)
    Bicubic = 2,
    /// Lanczos-3 3-lobe sinc-windowed sinc filtering (Lanczos, 1956; Turkowski, 1990)
    Lanczos3 = 3,
    /// Pixel area relation / box averaging with exact subpixel 2D bounding area overlap integration (Crow, 1984; Turkowski, 1990)
    Area = 4,
    /// Passthrough mode: copy source region to destination without interpolation
    Passthrough = 100,
}

/// Hardware backend providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum BackendType {
    Auto = 0,
    Vulkan = 1,
    OpenGL = 2,
    Npu = 3,
    Hw2d = 4,
    Cpu = 5,
    Passthrough = 6,
    OpenCL = 7,
}

/// Scalix error types.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ScalixError {
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Invalid stride {stride} bytes, minimum required is {min_stride} bytes")]
    InvalidStride { stride: usize, min_stride: usize },

    #[error("Buffer too small: required {required} bytes, got {actual} bytes")]
    BufferTooSmall { required: usize, actual: usize },

    #[error("Null pointer provided for required image buffer")]
    NullPointer,

    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),

    #[error("Requested backend unavailable: {0:?}")]
    BackendUnavailable(BackendType),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Unaligned memory pointer: address 0x{address:x} is not {alignment}-byte aligned")]
    UnalignedPointer { address: usize, alignment: usize },

    #[error("DMA allocator unavailable: {0}")]
    DmaUnavailable(String),

    #[error("DMA buffer allocation failed: {0}")]
    DmaAllocationFailed(String),

    #[error("DMA buffer memory map failed: {0}")]
    DmaMapFailed(String),

    #[error("DMA buffer sync failed: {0}")]
    DmaSyncFailed(String),

    #[error("Task timed out")]
    Timeout,

    #[error("Worker channel closed unexpectedly")]
    ChannelClosed,
}

pub type Result<T> = std::result::Result<T, ScalixError>;

/// Required memory alignment (64 bytes / cache-line & AVX-512 vector boundary) for image buffer pointers.
pub const REQUIRED_MEMORY_ALIGNMENT: usize = 64;

/// RAII heap buffer guaranteed to be 64-byte aligned for cache-line efficiency and SIMD vectorization.
#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    size: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates a zeroed 64-byte aligned memory buffer.
    pub fn new(size: usize) -> Result<Self> {
        Self::with_alignment(size, REQUIRED_MEMORY_ALIGNMENT)
    }

    /// Allocates a zeroed memory buffer with custom alignment.
    pub fn with_alignment(size: usize, alignment: usize) -> Result<Self> {
        let size = size.max(1);
        let layout = std::alloc::Layout::from_size_align(size, alignment)
            .map_err(|e| ScalixError::ExecutionFailed(format!("Invalid layout: {e}")))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(ScalixError::ExecutionFailed(
                "Failed to allocate aligned buffer".to_string(),
            ));
        }
        Ok(Self { ptr, layout, size })
    }

    /// Creates an AlignedBuffer by copying from a byte slice.
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        let mut buf = Self::new(slice.len())?;
        buf.as_mut_slice().copy_from_slice(slice);
        Ok(buf)
    }

    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice()).expect("Failed to clone AlignedBuffer")
    }
}

impl PartialEq for AlignedBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Read-only image descriptor for input sources.
#[derive(Debug, Clone)]
pub struct ImageDesc<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: &'a [u8],
    pub dma_buf_fd: Option<i32>,
}

impl<'a> ImageDesc<'a> {
    pub fn new(
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        data: &'a [u8],
    ) -> Result<Self> {
        let addr = data.as_ptr() as usize;
        if !addr.is_multiple_of(REQUIRED_MEMORY_ALIGNMENT) {
            return Err(ScalixError::UnalignedPointer {
                address: addr,
                alignment: REQUIRED_MEMORY_ALIGNMENT,
            });
        }
        let min_size = format.min_buffer_size(width, height, stride)?;
        if data.len() < min_size {
            return Err(ScalixError::BufferTooSmall {
                required: min_size,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            stride,
            format,
            data,
            dma_buf_fd: None,
        })
    }

    pub fn from_geometry(geom: ImageGeometry, data: &'a [u8]) -> Result<Self> {
        Self::new(
            geom.dimensions.width,
            geom.dimensions.height,
            geom.stride,
            geom.format,
            data,
        )
    }

    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        ImageDimensions::new(self.width, self.height)
    }

    #[inline]
    #[must_use]
    pub const fn with_dma_buf(mut self, fd: i32) -> Self {
        self.dma_buf_fd = Some(fd);
        self
    }
}

/// Mutable image descriptor for destination targets.
#[derive(Debug)]
pub struct ImageDescMut<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: &'a mut [u8],
    pub dma_buf_fd: Option<i32>,
}

impl<'a> ImageDescMut<'a> {
    pub fn new(
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        data: &'a mut [u8],
    ) -> Result<Self> {
        let addr = data.as_ptr() as usize;
        if !addr.is_multiple_of(REQUIRED_MEMORY_ALIGNMENT) {
            return Err(ScalixError::UnalignedPointer {
                address: addr,
                alignment: REQUIRED_MEMORY_ALIGNMENT,
            });
        }
        let min_size = format.min_buffer_size(width, height, stride)?;
        if data.len() < min_size {
            return Err(ScalixError::BufferTooSmall {
                required: min_size,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            stride,
            format,
            data,
            dma_buf_fd: None,
        })
    }

    pub fn from_geometry(geom: ImageGeometry, data: &'a mut [u8]) -> Result<Self> {
        Self::new(
            geom.dimensions.width,
            geom.dimensions.height,
            geom.stride,
            geom.format,
            data,
        )
    }

    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        ImageDimensions::new(self.width, self.height)
    }

    #[inline]
    #[must_use]
    pub const fn with_dma_buf(mut self, fd: i32) -> Self {
        self.dma_buf_fd = Some(fd);
        self
    }
}

pub use crate::dma::DmaAllocatorType;

/// Backend-specific execution options and strategy metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendOptions {
    #[default]
    None,
    Vulkan(crate::backend::vulkan::VulkanOptions),
    Gl(crate::backend::gl::GlOptions),
    OpenCl(crate::backend::opencl::OpenClOptions),
}

/// Dynamic per-resize operation metadata and configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeOptions {
    pub filter: FilterMode,
    pub backend_options: BackendOptions,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            filter: FilterMode::Bilinear,
            backend_options: BackendOptions::None,
        }
    }
}

impl From<FilterMode> for ResizeOptions {
    fn from(filter: FilterMode) -> Self {
        Self {
            filter,
            backend_options: BackendOptions::None,
        }
    }
}

impl ResizeOptions {
    #[inline]
    #[must_use]
    pub fn new(filter: FilterMode) -> Self {
        Self {
            filter,
            backend_options: BackendOptions::None,
        }
    }

    #[inline]
    #[must_use]
    pub fn with_vulkan_options(mut self, opts: crate::backend::vulkan::VulkanOptions) -> Self {
        self.backend_options = BackendOptions::Vulkan(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_vulkan_strategy(
        mut self,
        strategy: crate::backend::vulkan::VulkanStrategy,
    ) -> Self {
        let mut opts = match self.backend_options {
            BackendOptions::Vulkan(v) => v,
            _ => crate::backend::vulkan::VulkanOptions::default(),
        };
        opts.strategy = strategy;
        self.backend_options = BackendOptions::Vulkan(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_gl_options(mut self, opts: crate::backend::gl::GlOptions) -> Self {
        self.backend_options = BackendOptions::Gl(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_gl_strategy(mut self, strategy: crate::backend::gl::GlStrategy) -> Self {
        let mut opts = match self.backend_options {
            BackendOptions::Gl(g) => g,
            _ => crate::backend::gl::GlOptions::default(),
        };
        opts.strategy = strategy;
        self.backend_options = BackendOptions::Gl(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_opencl_options(mut self, opts: crate::backend::opencl::OpenClOptions) -> Self {
        self.backend_options = BackendOptions::OpenCl(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_opencl_strategy(
        mut self,
        strategy: crate::backend::opencl::OpenClStrategy,
    ) -> Self {
        let mut opts = match self.backend_options {
            BackendOptions::OpenCl(c) => c,
            _ => crate::backend::opencl::OpenClOptions::default(),
        };
        opts.strategy = strategy;
        self.backend_options = BackendOptions::OpenCl(opts);
        self
    }

    #[inline]
    #[must_use]
    pub fn with_max_mip_levels(mut self, max_levels: u32) -> Self {
        match self.backend_options {
            BackendOptions::Vulkan(mut v) => {
                v.max_mip_levels = max_levels;
                self.backend_options = BackendOptions::Vulkan(v);
            }
            BackendOptions::Gl(mut g) => {
                g.max_mip_levels = max_levels;
                self.backend_options = BackendOptions::Gl(g);
            }
            BackendOptions::OpenCl(_) | BackendOptions::None => {
                self.backend_options =
                    BackendOptions::Vulkan(crate::backend::vulkan::VulkanOptions {
                        strategy: crate::backend::vulkan::VulkanStrategy::Auto,
                        max_mip_levels: max_levels,
                    });
            }
        }
        self
    }

    /// Extracts Vulkan options or returns default.
    #[inline]
    #[must_use]
    pub fn vulkan_options(&self) -> crate::backend::vulkan::VulkanOptions {
        match self.backend_options {
            BackendOptions::Vulkan(v) => v,
            _ => crate::backend::vulkan::VulkanOptions::default(),
        }
    }

    /// Extracts OpenGL options or returns default.
    #[inline]
    #[must_use]
    pub fn gl_options(&self) -> crate::backend::gl::GlOptions {
        match self.backend_options {
            BackendOptions::Gl(g) => g,
            _ => crate::backend::gl::GlOptions::default(),
        }
    }

    /// Extracts OpenCL options or returns default.
    #[inline]
    #[must_use]
    pub fn opencl_options(&self) -> crate::backend::opencl::OpenClOptions {
        match self.backend_options {
            BackendOptions::OpenCl(c) => c,
            _ => crate::backend::opencl::OpenClOptions::default(),
        }
    }
}
