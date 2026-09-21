use crate::types::{BackendType, FilterMode, ImageDesc, ImageDescMut, Result, ScalixError};

/// Common hardware/software backend provider interface.
pub trait Backend: Send + Sync {
    /// Returns human-readable name of the backend.
    fn name(&self) -> &'static str;

    /// Returns the backend type enumeration.
    fn backend_type(&self) -> BackendType;

    /// Checks if this backend is supported and initialized on the current system.
    fn is_available(&self) -> bool;

    /// Executes image processing / resize from source to destination.
    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, filter: FilterMode) -> Result<()>;
}

/// A reference passthrough backend that performs direct memory transfer
/// without resizing (or clips/pads according to dimensions).
/// Used for memory transfer verification, benchmarking baseline latency, and testing.
#[derive(Debug, Default, Clone)]
pub struct PassthroughBackend;

impl PassthroughBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for PassthroughBackend {
    fn name(&self) -> &'static str {
        "Passthrough / Memory Transfer Backend"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Passthrough
    }

    fn is_available(&self) -> bool {
        true
    }

    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, _filter: FilterMode) -> Result<()> {
        if src.format != dst.format {
            return Err(ScalixError::ExecutionFailed(format!(
                "Format conversion not supported in passthrough mode (src: {:?}, dst: {:?})",
                src.format, dst.format
            )));
        }

        let copy_width = src.width.min(dst.width);
        let copy_height = src.height.min(dst.height);

        let bpp = src
            .format
            .bytes_per_pixel()
            .ok_or_else(|| ScalixError::UnsupportedFormat(src.format))?;

        let row_bytes = (copy_width as usize) * bpp;

        for y in 0..(copy_height as usize) {
            let src_offset = y * src.stride;
            let dst_offset = y * dst.stride;

            let src_row = &src.data[src_offset..src_offset + row_bytes];
            let dst_row = &mut dst.data[dst_offset..dst_offset + row_bytes];

            dst_row.copy_from_slice(src_row);
        }

        Ok(())
    }
}
