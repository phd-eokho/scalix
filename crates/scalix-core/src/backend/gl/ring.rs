//! OpenGL Texture & Framebuffer Cache
//!
//! Each slot caches a source/destination texture pair and their FBOs so repeated resizes of
//! the same geometry skip reallocation.
//!
//! NOTE: Every resize is synchronous: the upload copies from client memory and the readback is
//! a blocking `glReadPixels`, so slots never overlap GPU work. Action: add PBO-backed async
//! readback only together with an asynchronous API that can return before the GPU finishes.

use crate::backend::gl::context::*;
use crate::types::{ImageDesc, ImageDescMut, PixelFormat, Result, ScalixError};
use std::ffi::c_void;
use std::sync::Arc;

pub const DEFAULT_GL_RING_SLOTS: usize = 3;

/// An isolated in-flight execution slot in the OpenGL staging ring.
pub struct GlStagingSlot {
    /// Cached persistent source texture (reused across frames)
    pub src_texture: u32,
    pub src_tex_w: u32,
    pub src_tex_h: u32,
    pub src_tex_internal: u32,
    pub src_tex_format: u32,
    pub src_tex_type: u32,

    /// Cached persistent destination texture (reused across frames)
    pub dst_texture: u32,
    pub dst_tex_w: u32,
    pub dst_tex_h: u32,
    pub dst_tex_internal: u32,
    pub dst_tex_format: u32,
    pub dst_tex_type: u32,

    /// Framebuffer Objects for Blit/Raster/ReadPixels
    pub src_fbo: u32,
    pub dst_fbo: u32,
}

impl GlStagingSlot {
    pub fn new(ctx: &Arc<EglContext>) -> Result<Self> {
        let gl = &ctx.gl;
        let mut textures = [0u32; 2];
        let mut fbos = [0u32; 2];

        unsafe {
            (gl.glGenTextures)(2, textures.as_mut_ptr());
            (gl.glGenFramebuffers)(2, fbos.as_mut_ptr());
        }

        Ok(Self {
            src_texture: textures[0],
            src_tex_w: 0,
            src_tex_h: 0,
            src_tex_internal: 0,
            src_tex_format: 0,
            src_tex_type: 0,
            dst_texture: textures[1],
            dst_tex_w: 0,
            dst_tex_h: 0,
            dst_tex_internal: 0,
            dst_tex_format: 0,
            dst_tex_type: 0,
            src_fbo: fbos[0],
            dst_fbo: fbos[1],
        })
    }

    /// Ensures source texture allocation is configured with matching dimensions and format.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_src_texture(
        &mut self,
        ctx: &Arc<EglContext>,
        w: u32,
        h: u32,
        internal: u32,
        format: u32,
        typ: u32,
        gl_filter: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        let needs_realloc = self.src_tex_w != w
            || self.src_tex_h != h
            || self.src_tex_internal != internal
            || self.src_tex_format != format
            || self.src_tex_type != typ;

        unsafe {
            (gl.glBindTexture)(GL_TEXTURE_2D, self.src_texture);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gl_filter as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gl_filter as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

            if needs_realloc {
                (gl.glTexImage2D)(
                    GL_TEXTURE_2D,
                    0,
                    internal as i32,
                    w as i32,
                    h as i32,
                    0,
                    format,
                    typ,
                    std::ptr::null(),
                );
                self.src_tex_w = w;
                self.src_tex_h = h;
                self.src_tex_internal = internal;
                self.src_tex_format = format;
                self.src_tex_type = typ;
            }
        }
        Ok(self.src_texture)
    }

    /// Ensures destination texture allocation is configured with matching dimensions and format.
    ///
    /// With `glTexStorage2D` available the texture is immutable, which GLES requires for
    /// `glBindImageTexture` (compute). Immutable storage cannot be respecified, so a new texture
    /// name is created on reallocation; callers attach the returned name every call.
    pub fn ensure_dst_texture(
        &mut self,
        ctx: &Arc<EglContext>,
        w: u32,
        h: u32,
        internal: u32,
        format: u32,
        typ: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        let needs_realloc = self.dst_tex_w != w
            || self.dst_tex_h != h
            || self.dst_tex_internal != internal
            || self.dst_tex_format != format
            || self.dst_tex_type != typ;

        let tex_storage = gl.glTexStorage2D.filter(|_| ctx.has_tex_storage);
        unsafe {
            if needs_realloc && tex_storage.is_some() {
                (gl.glDeleteTextures)(1, &self.dst_texture);
                (gl.glGenTextures)(1, &mut self.dst_texture);
            }
            (gl.glBindTexture)(GL_TEXTURE_2D, self.dst_texture);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE as i32);
            (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

            if needs_realloc {
                match tex_storage {
                    Some(storage) => storage(GL_TEXTURE_2D, 1, internal, w as i32, h as i32),
                    None => (gl.glTexImage2D)(
                        GL_TEXTURE_2D,
                        0,
                        internal as i32,
                        w as i32,
                        h as i32,
                        0,
                        format,
                        typ,
                        std::ptr::null(),
                    ),
                }
                self.dst_tex_w = w;
                self.dst_tex_h = h;
                self.dst_tex_internal = internal;
                self.dst_tex_format = format;
                self.dst_tex_type = typ;
            }
        }
        Ok(self.dst_texture)
    }

    /// Uploads source pixels straight from client memory, honoring `src.stride`.
    pub fn upload_src_data(
        &mut self,
        ctx: &Arc<EglContext>,
        src: &ImageDesc,
        internal: u32,
        format: u32,
        typ: u32,
        gl_filter: u32,
    ) -> Result<u32> {
        let gl = &ctx.gl;
        let (w, h) = (src.width, src.height);
        let row_length = row_length_px(src.format, src.width, src.stride)?;
        self.ensure_src_texture(ctx, w, h, internal, format, typ, gl_filter)?;

        unsafe {
            (gl.glBindTexture)(GL_TEXTURE_2D, self.src_texture);
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            match row_length {
                Some(len) => {
                    (gl.glPixelStorei)(GL_UNPACK_ROW_LENGTH, len);
                    (gl.glTexSubImage2D)(
                        GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        w as i32,
                        h as i32,
                        format,
                        typ,
                        src.data.as_ptr() as *const c_void,
                    );
                    (gl.glPixelStorei)(GL_UNPACK_ROW_LENGTH, 0);
                }
                // Stride is not a whole number of pixels: GL cannot express it, upload per row.
                None => {
                    for (y, row) in src.data.chunks(src.stride).take(h as usize).enumerate() {
                        (gl.glTexSubImage2D)(
                            GL_TEXTURE_2D,
                            0,
                            0,
                            y as i32,
                            w as i32,
                            1,
                            format,
                            typ,
                            row.as_ptr() as *const c_void,
                        );
                    }
                }
            }
        }

        Ok(self.src_texture)
    }
}

/// Returns the GL row length (in pixels) for `stride`, or `None` when the stride is not a
/// whole number of pixels and rows must be transferred individually.
fn row_length_px(
    format: crate::types::PixelFormat,
    width: u32,
    stride: usize,
) -> Result<Option<i32>> {
    let bpp = format
        .bytes_per_pixel()
        .ok_or(ScalixError::UnsupportedFormat(format))?;
    let min_stride = width as usize * bpp;
    if stride < min_stride {
        return Err(ScalixError::InvalidStride { stride, min_stride });
    }
    if !stride.is_multiple_of(bpp) {
        return Ok(None);
    }
    let len = i32::try_from(stride / bpp)
        .map_err(|_| ScalixError::InvalidStride { stride, min_stride })?;
    Ok(Some(len))
}

/// Reads the bound `GL_READ_FRAMEBUFFER` into `dst`, honoring `dst.stride`.
///
/// `src_format` is the format the framebuffer contents were produced from; on GLES it decides
/// whether red and blue must be swapped (see [`super::blit::get_gl_format_tuple`]).
///
/// # Safety
/// The context must be current and a complete read framebuffer of at least
/// `dst.width x dst.height` must be bound.
pub(crate) unsafe fn read_pixels_into(
    ctx: &EglContext,
    dst: &mut ImageDescMut,
    format: u32,
    typ: u32,
    src_format: PixelFormat,
) -> Result<()> {
    let gl = &ctx.gl;
    let swap_rb = is_bgr_ordered(src_format) != is_bgr_ordered(dst.format);
    // GLES only guarantees GL_RGBA/GL_UNSIGNED_BYTE readback: read that and repack on the CPU
    // unless the bytes already land in `dst` order.
    if ctx.is_gles && (dst.format.bytes_per_pixel() != Some(4) || swap_rb) {
        return read_rgba_and_repack(gl, dst, swap_rb);
    }
    let (format, typ) = if ctx.is_gles {
        (GL_RGBA, GL_UNSIGNED_BYTE)
    } else {
        (format, typ)
    };
    let (w, h) = (dst.width as i32, dst.height as i32);
    let row_length = row_length_px(dst.format, dst.width, dst.stride)?;
    (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);
    match row_length {
        Some(len) => {
            (gl.glPixelStorei)(GL_PACK_ROW_LENGTH, len);
            (gl.glReadPixels)(
                0,
                0,
                w,
                h,
                format,
                typ,
                dst.data.as_mut_ptr() as *mut c_void,
            );
            (gl.glPixelStorei)(GL_PACK_ROW_LENGTH, 0);
        }
        None => {
            for (y, row) in dst.data.chunks_mut(dst.stride).take(h as usize).enumerate() {
                (gl.glReadPixels)(
                    0,
                    y as i32,
                    w,
                    1,
                    format,
                    typ,
                    row.as_mut_ptr() as *mut c_void,
                );
            }
        }
    }
    Ok(())
}

fn is_bgr_ordered(format: PixelFormat) -> bool {
    matches!(format, PixelFormat::Bgr888 | PixelFormat::Bgra8888)
}

/// GLES readback path: reads tightly packed RGBA8 and converts it into `dst`.
///
/// # Safety
/// Same contract as [`read_pixels_into`].
unsafe fn read_rgba_and_repack(
    gl: &GlFunctions,
    dst: &mut ImageDescMut,
    swap_rb: bool,
) -> Result<()> {
    let (w, h) = (dst.width as usize, dst.height as usize);
    let mut rgba = vec![0u8; w * h * 4];
    (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);
    (gl.glPixelStorei)(GL_PACK_ROW_LENGTH, 0);
    (gl.glReadPixels)(
        0,
        0,
        w as i32,
        h as i32,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        rgba.as_mut_ptr() as *mut c_void,
    );
    repack_rgba(&rgba, dst, swap_rb)
}

/// Converts tightly packed RGBA8 rows into `dst` (3 or 4 bytes per pixel), optionally swapping
/// red and blue.
fn repack_rgba(rgba: &[u8], dst: &mut ImageDescMut, swap_rb: bool) -> Result<()> {
    let bpp = dst
        .format
        .bytes_per_pixel()
        .filter(|&b| b == 3 || b == 4)
        .ok_or(ScalixError::UnsupportedFormat(dst.format))?;
    let w = dst.width as usize;
    let rows = rgba
        .chunks_exact(w * 4)
        .zip(dst.data.chunks_mut(dst.stride));
    for (src_row, dst_row) in rows.take(dst.height as usize) {
        for (s, d) in src_row
            .chunks_exact(4)
            .zip(dst_row[..w * bpp].chunks_exact_mut(bpp))
        {
            let (r, b) = if swap_rb { (s[2], s[0]) } else { (s[0], s[2]) };
            d[0] = r;
            d[1] = s[1];
            d[2] = b;
            if bpp == 4 {
                d[3] = s[3];
            }
        }
    }
    Ok(())
}

unsafe impl Send for GlStagingSlot {}
unsafe impl Sync for GlStagingSlot {}

/// Round-robin cache of [`GlStagingSlot`]s for OpenGL/GLES.
pub struct GlStagingRing {
    ctx: Arc<EglContext>,
    slots: Vec<GlStagingSlot>,
    current_idx: usize,
}

unsafe impl Send for GlStagingRing {}
unsafe impl Sync for GlStagingRing {}

impl GlStagingRing {
    /// Creates a new OpenGL staging ring with `slot_count` slots (default: 3).
    pub fn new(ctx: Arc<EglContext>, slot_count: usize) -> Result<Self> {
        let count = slot_count.max(1);
        let mut slots = Vec::with_capacity(count);

        {
            let _guard = ctx.bind_current()?;
            for _ in 0..count {
                slots.push(GlStagingSlot::new(&ctx)?);
            }
        }

        Ok(Self {
            ctx,
            slots,
            current_idx: 0,
        })
    }

    /// Returns the next slot in round-robin order.
    pub fn acquire_slot(&mut self) -> Result<(usize, &mut GlStagingSlot)> {
        let idx = self.current_idx;
        self.current_idx = (self.current_idx + 1) % self.slots.len();
        Ok((idx, &mut self.slots[idx]))
    }
}

impl Drop for GlStagingRing {
    fn drop(&mut self) {
        if let Ok(_guard) = self.ctx.bind_current() {
            let gl = &self.ctx.gl;
            for slot in &self.slots {
                unsafe {
                    let textures = [slot.src_texture, slot.dst_texture];
                    let fbos = [slot.src_fbo, slot.dst_fbo];

                    (gl.glDeleteTextures)(2, textures.as_ptr());
                    (gl.glDeleteFramebuffers)(2, fbos.as_ptr());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repack(dst_format: PixelFormat, swap_rb: bool) -> Vec<u8> {
        // 2x2 RGBA source, distinct value per channel.
        let rgba: Vec<u8> = (0..16).collect();
        let bpp = dst_format.bytes_per_pixel().unwrap();
        let stride = 2 * bpp + 1; // padded row
        let mut out = crate::types::AlignedBuffer::new(stride * 2).unwrap();
        out.as_mut_slice().fill(0xEE);
        let mut dst = ImageDescMut::new(2, 2, stride, dst_format, out.as_mut_slice()).unwrap();
        repack_rgba(&rgba, &mut dst, swap_rb).unwrap();
        out.as_slice().to_vec()
    }

    #[test]
    fn repack_drops_alpha_for_three_channel_dst() {
        assert_eq!(
            repack(PixelFormat::Rgb888, false),
            [0, 1, 2, 4, 5, 6, 0xEE, 8, 9, 10, 12, 13, 14, 0xEE]
        );
    }

    #[test]
    fn repack_swaps_red_and_blue() {
        assert_eq!(
            repack(PixelFormat::Bgr888, true),
            [2, 1, 0, 6, 5, 4, 0xEE, 10, 9, 8, 14, 13, 12, 0xEE]
        );
        assert_eq!(
            repack(PixelFormat::Bgra8888, true),
            [2, 1, 0, 3, 6, 5, 4, 7, 0xEE, 10, 9, 8, 11, 14, 13, 12, 15, 0xEE]
        );
    }
}
