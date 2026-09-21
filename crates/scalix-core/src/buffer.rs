use crate::types::{ImageDesc, ImageDescMut, PixelFormat, Result, ScalixError};

/// An owned heap-allocated image buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedImage {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

impl OwnedImage {
    /// Allocates an uninitialized/zeroed image buffer for the specified dimensions and format.
    pub fn allocate(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        let stride = format.min_stride(width)?;
        let size = format.min_buffer_size(width, height, stride)?;
        Ok(Self {
            width,
            height,
            stride,
            format,
            data: vec![0u8; size],
        })
    }

    /// Creates an OwnedImage from an existing vector of data with dimension/stride validation.
    pub fn from_vec(
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        data: Vec<u8>,
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
        })
    }

    /// Borrows this image as a read-only ImageDesc.
    pub fn as_desc(&self) -> ImageDesc<'_> {
        ImageDesc {
            width: self.width,
            height: self.height,
            stride: self.stride,
            format: self.format,
            data: &self.data,
            dma_buf_fd: None,
        }
    }

    /// Borrows this image as a mutable ImageDescMut.
    pub fn as_desc_mut(&mut self) -> ImageDescMut<'_> {
        ImageDescMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            format: self.format,
            data: &mut self.data,
            dma_buf_fd: None,
        }
    }
}
