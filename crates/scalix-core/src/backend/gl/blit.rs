//! OpenGL FBO Blit Hardware Scaler (`glBlitFramebuffer`)

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::context::*;
use super::ring::GlStagingRing;
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, PixelFormat, Result, ScalixError};

pub struct GlBlitter {
    ctx: Arc<EglContext>,
    ring: Arc<Mutex<GlStagingRing>>,
    profiler: Arc<dyn Profiler>,
}

impl GlBlitter {
    pub fn new(
        ctx: Arc<EglContext>,
        ring: Arc<Mutex<GlStagingRing>>,
        profiler: Arc<dyn Profiler>,
    ) -> Self {
        Self {
            ctx,
            ring,
            profiler,
        }
    }

    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        let wall_start = Instant::now();
        let _guard = self.ctx.bind_current()?;
        // GL queues work asynchronously; finishing each phase attributes GPU time to it.
        let profiling = self.profiler.is_enabled();
        let gl = &self.ctx.gl;

        let (src_internal, src_format, src_type) =
            get_gl_format_tuple(src.format, self.ctx.is_gles)?;
        let (dst_internal, dst_format, dst_type) =
            get_gl_format_tuple(dst.format, self.ctx.is_gles)?;

        let gl_filter = match filter {
            FilterMode::Nearest => GL_NEAREST,
            _ => GL_LINEAR,
        };

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire GlStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        unsafe {
            // Unpack alignment 1 for tightly packed RGB/BGR
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);

            // 1. Setup Source Texture & Upload from client memory
            let upload_start = Instant::now();
            let src_tex = slot.upload_src_data(
                &self.ctx,
                src,
                src_internal,
                src_format,
                src_type,
                gl_filter,
            )?;

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.src_fbo);
            (gl.glFramebufferTexture2D)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                src_tex,
                0,
            );

            // 2. Setup Destination Texture & FBO
            let dst_tex = slot.ensure_dst_texture(
                &self.ctx,
                dst.width,
                dst.height,
                dst_internal,
                dst_format,
                dst_type,
            )?;

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.dst_fbo);
            (gl.glFramebufferTexture2D)(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                dst_tex,
                0,
            );
            if profiling {
                (gl.glFinish)();
            }
            let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

            // 3. Blit Framebuffer
            let blit_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.src_fbo);
            (gl.glBindFramebuffer)(GL_DRAW_FRAMEBUFFER, slot.dst_fbo);
            (gl.glBlitFramebuffer)(
                0,
                0,
                src.width as i32,
                src.height as i32,
                0,
                0,
                dst.width as i32,
                dst.height as i32,
                GL_COLOR_BUFFER_BIT,
                gl_filter,
            );
            if profiling {
                (gl.glFinish)();
            }
            let blit_ms = blit_start.elapsed().as_secs_f64() * 1000.0;

            // 4. Download Pixels
            let download_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.dst_fbo);
            super::ring::read_pixels_into(&self.ctx, dst, dst_format, dst_type, src.format)?;
            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
            let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

            self.ctx.check_gl_error("blit")?;

            let total_wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            self.profiler.record(ProfileMetrics {
                host_unpack_ms: 0.0,
                gpu_upload_ms: upload_ms,
                gpu_pure_blit_ms: blit_ms,
                gpu_download_ms: download_ms,
                host_repack_ms: 0.0,
                driver_sync_ms: 0.0,
                total_wall_ms,
            });
        }

        Ok(())
    }
}

/// Returns `(internal format, transfer format, type)` for `format`.
///
/// GLES has no `GL_BGR`/`GL_BGRA`, so there BGR-ordered pixels are transferred as RGB(A) bytes
/// unchanged: every resize treats channels independently, and the readback restores the order
/// (see [`super::ring::read_pixels_into`]).
pub(crate) fn get_gl_format_tuple(format: PixelFormat, is_gles: bool) -> Result<(u32, u32, u32)> {
    match (format, is_gles) {
        (PixelFormat::Rgba8888, _) | (PixelFormat::Bgra8888, true) => {
            Ok((GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE))
        }
        (PixelFormat::Bgra8888, false) => Ok((GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE)),
        (PixelFormat::Rgb888, _) | (PixelFormat::Bgr888, true) => {
            Ok((GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE))
        }
        (PixelFormat::Bgr888, false) => Ok((GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE)),
        (other, _) => Err(ScalixError::UnsupportedFormat(other)),
    }
}
