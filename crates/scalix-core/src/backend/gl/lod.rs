use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::blit::get_gl_format_tuple;
use super::context::*;
use super::ring::GlStagingRing;
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{ImageDesc, ImageDescMut, ResizeOptions, Result, ScalixError};

pub struct GlLodDownscaler {
    ctx: Arc<EglContext>,
    ring: Arc<Mutex<GlStagingRing>>,
    profiler: Arc<dyn Profiler>,
}

impl GlLodDownscaler {
    pub fn new(
        ctx: Arc<EglContext>,
        ring: Arc<Mutex<GlStagingRing>>,
        profiler: Arc<dyn Profiler>,
    ) -> Self {
        Self { ctx, ring, profiler }
    }

    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()> {
        let wall_start = Instant::now();
        let _guard = self.ctx.bind_current()?;
        let gl = &self.ctx.gl;

        let (src_internal, src_format, src_type) = get_gl_format_tuple(src.format)?;
        let (dst_internal, dst_format, dst_type) = get_gl_format_tuple(dst.format)?;

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire GlStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        unsafe {
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);

            let upload_start = Instant::now();

            let ratio = if dst.width == 0 || dst.height == 0 {
                1
            } else {
                (src.width / dst.width).min(src.height / dst.height)
            };
            let num_levels = if ratio > 1 { ratio.ilog2() } else { 0 };
            let max_mip_levels = options.gl_options().max_mip_levels;
            let chosen_level = if max_mip_levels > 0 {
                num_levels.min(max_mip_levels)
            } else {
                num_levels
            };

            // Upload source to slot.src_texture via PBO DMA staging
            let src_tex = slot.upload_src_data(
                &self.ctx,
                src.data,
                src.width,
                src.height,
                src_internal,
                src_format,
                src_type,
                GL_LINEAR,
            )?;

            // Generate hardware mipchain pyramid if downscaling across multiple levels
            if chosen_level > 0 {
                let max_level = (src.width.max(src.height) as f32).log2().floor() as u32;
                (gl.glBindTexture)(GL_TEXTURE_2D, src_tex);
                (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, max_level as i32);
                (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR as i32);
                (gl.glGenerateMipmap)(GL_TEXTURE_2D);
            }

            let mip_w = (src.width >> chosen_level).max(1);
            let mip_h = (src.height >> chosen_level).max(1);

            // Setup destination texture and FBO
            let dst_tex = slot.ensure_dst_texture(
                &self.ctx,
                dst.width,
                dst.height,
                dst_internal,
                dst_format,
                dst_type,
            )?;

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.dst_fbo);
            (gl.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);
            let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

            let blit_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.src_fbo);
            (gl.glFramebufferTexture2D)(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, chosen_level as i32);
            (gl.glBindFramebuffer)(GL_DRAW_FRAMEBUFFER, slot.dst_fbo);
            (gl.glBlitFramebuffer)(
                0, 0, mip_w as i32, mip_h as i32,
                0, 0, dst.width as i32, dst.height as i32,
                GL_COLOR_BUFFER_BIT,
                GL_LINEAR,
            );

            slot.signal_fence(&self.ctx);
            let blit_ms = blit_start.elapsed().as_secs_f64() * 1000.0;

            // Download final output
            let download_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.dst_fbo);
            (gl.glReadPixels)(
                0, 0, dst.width as i32, dst.height as i32,
                dst_format, dst_type, dst.data.as_mut_ptr() as *mut std::ffi::c_void,
            );

            // Clean up / restore FBO attachments and texture states
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, slot.src_fbo);
            (gl.glFramebufferTexture2D)(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, 0);
            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
            if chosen_level > 0 {
                (gl.glBindTexture)(GL_TEXTURE_2D, src_tex);
                (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
                (gl.glTexParameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR as i32);
            }
            let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;

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
