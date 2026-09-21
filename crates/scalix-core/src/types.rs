use thiserror::Error;

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
    pub fn min_stride(self, width: u32) -> Result<usize> {
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
    pub fn min_buffer_size(self, width: u32, height: u32, stride: usize) -> Result<usize> {
        if width == 0 || height == 0 {
            return Err(ScalixError::InvalidDimensions { width, height });
        }
        let min_s = self.min_stride(width)?;
        if stride < min_s {
            return Err(ScalixError::InvalidStride {
                stride,
                min_stride: min_s,
            });
        }

        match self {
            PixelFormat::Nv12 | PixelFormat::Yuv420p => {
                // Y plane (stride * height) + UV planes (stride * height / 2)
                let y_size = stride
                    .checked_mul(height as usize)
                    .ok_or(ScalixError::InvalidDimensions { width, height })?;
                let uv_size = y_size / 2;
                Ok(y_size + uv_size)
            }
            _ => {
                let h = height as usize;
                // Last row needs at least min_stride bytes, preceding rows need stride bytes
                let preceding = stride
                    .checked_mul(h.saturating_sub(1))
                    .ok_or(ScalixError::InvalidDimensions { width, height })?;
                Ok(preceding + min_s)
            }
        }
    }
}

/// Image scaling filter and interpolation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum FilterMode {
    Nearest = 0,
    Bilinear = 1,
    Bicubic = 2,
    Lanczos3 = 3,
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

    pub fn with_dma_buf(mut self, fd: i32) -> Self {
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

    pub fn with_dma_buf(mut self, fd: i32) -> Self {
        self.dma_buf_fd = Some(fd);
        self
    }
}

/// Dynamic per-resize operation metadata and configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ResizeOptions {
    pub filter: FilterMode,
    pub vulkan: crate::backend::vulkan::VulkanOptions,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            filter: FilterMode::Bilinear,
            vulkan: crate::backend::vulkan::VulkanOptions::default(),
        }
    }
}

impl From<FilterMode> for ResizeOptions {
    fn from(filter: FilterMode) -> Self {
        Self {
            filter,
            vulkan: crate::backend::vulkan::VulkanOptions::default(),
        }
    }
}

impl ResizeOptions {
    pub fn new(filter: FilterMode) -> Self {
        Self {
            filter,
            vulkan: crate::backend::vulkan::VulkanOptions::default(),
        }
    }

    pub fn with_vulkan_strategy(mut self, strategy: crate::backend::vulkan::VulkanStrategy) -> Self {
        self.vulkan.strategy = strategy;
        self
    }
}
