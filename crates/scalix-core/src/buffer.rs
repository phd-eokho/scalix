use crate::types::{
    AlignedBuffer, ImageDesc, ImageDescMut, ImageDimensions, ImageGeometry, PixelFormat, Result,
    ScalixError,
};

/// An owned heap-allocated image buffer with guaranteed 64-byte alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedImage {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: AlignedBuffer,
}

impl OwnedImage {
    /// Allocates a zeroed 64-byte aligned image buffer for the specified dimensions and format.
    pub fn allocate(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        let stride = format.min_stride(width)?;
        let size = format.min_buffer_size(width, height, stride)?;
        Ok(Self {
            width,
            height,
            stride,
            format,
            data: AlignedBuffer::new(size)?,
        })
    }

    /// Allocates a zeroed 64-byte aligned image buffer for the specified geometry.
    pub fn allocate_geometry(geom: ImageGeometry) -> Result<Self> {
        let size = geom.min_buffer_size()?;
        Ok(Self {
            width: geom.dimensions.width,
            height: geom.dimensions.height,
            stride: geom.stride,
            format: geom.format,
            data: AlignedBuffer::new(size)?,
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
        Self::from_slice(width, height, stride, format, &data)
    }

    /// Creates an OwnedImage from a byte slice with 64-byte alignment.
    pub fn from_slice(
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        slice: &[u8],
    ) -> Result<Self> {
        let min_size = format.min_buffer_size(width, height, stride)?;
        if slice.len() < min_size {
            return Err(ScalixError::BufferTooSmall {
                required: min_size,
                actual: slice.len(),
            });
        }
        let data = AlignedBuffer::from_slice(slice)?;
        Ok(Self {
            width,
            height,
            stride,
            format,
            data,
        })
    }

    /// Creates an OwnedImage from geometry and an existing vector of data.
    pub fn from_geometry_vec(geom: ImageGeometry, data: Vec<u8>) -> Result<Self> {
        Self::from_vec(
            geom.dimensions.width,
            geom.dimensions.height,
            geom.stride,
            geom.format,
            data,
        )
    }

    /// Returns the spatial dimensions of the image.
    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> ImageDimensions {
        ImageDimensions::new(self.width, self.height)
    }

    /// Returns the byte stride between consecutive rows.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the pixel format.
    #[inline]
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Returns the geometry of the image buffer.
    #[inline]
    #[must_use]
    pub const fn geometry(&self) -> ImageGeometry {
        ImageGeometry {
            dimensions: self.dimensions(),
            stride: self.stride,
            format: self.format,
        }
    }

    /// Returns a slice of the raw pixel data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable slice of the raw pixel data.
    #[inline]
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Borrows this image as a read-only ImageDesc.
    #[inline]
    #[must_use]
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
    #[inline]
    #[must_use]
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
