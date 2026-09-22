use super::Backend;
use crate::types::{
    BackendType, ImageDesc, ImageDescMut, ImageDimensions, ResizeOptions, Result, ScalixError,
};

/// A reference passthrough backend that performs direct memory transfer
/// without resizing (or clips/pads according to dimensions).
#[derive(Debug, Default, Clone)]
pub struct PassthroughBackend;

impl PassthroughBackend {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Backend for PassthroughBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "Passthrough / Memory Transfer Backend"
    }

    #[inline]
    fn backend_type(&self) -> BackendType {
        BackendType::Passthrough
    }

    #[inline]
    fn is_available(&self) -> bool {
        true
    }

    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        _options: &ResizeOptions,
    ) -> Result<()> {
        if src.format != dst.format {
            return Err(ScalixError::ExecutionFailed(format!(
                "Format conversion not supported in passthrough mode (src: {:?}, dst: {:?})",
                src.format, dst.format
            )));
        }

        let src_dims = src.dimensions();
        let dst_dims = dst.dimensions();
        let copy_dims = ImageDimensions::new(
            src_dims.width.min(dst_dims.width),
            src_dims.height.min(dst_dims.height),
        );

        let bpp = src
            .format
            .bytes_per_pixel()
            .ok_or(ScalixError::UnsupportedFormat(src.format))?;

        let row_bytes = (copy_dims.width as usize).saturating_mul(bpp);

        // Fast contiguous path: if strides match row_bytes and heights match
        if src.stride == row_bytes
            && dst.stride == row_bytes
            && copy_dims.height == src_dims.height
            && copy_dims.height == dst_dims.height
        {
            let total_bytes = (copy_dims.height as usize).saturating_mul(row_bytes);
            dst.data[..total_bytes].copy_from_slice(&src.data[..total_bytes]);
            return Ok(());
        }

        for y in 0..(copy_dims.height as usize) {
            let src_offset = y.saturating_mul(src.stride);
            let dst_offset = y.saturating_mul(dst.stride);

            let src_row = &src.data[src_offset..src_offset.saturating_add(row_bytes)];
            let dst_row = &mut dst.data[dst_offset..dst_offset.saturating_add(row_bytes)];

            dst_row.copy_from_slice(src_row);
        }

        Ok(())
    }
}
