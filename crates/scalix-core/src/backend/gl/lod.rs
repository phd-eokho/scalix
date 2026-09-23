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

        let max_mip_levels = options.gl_options().max_mip_levels;
        let max_passes = if max_mip_levels > 0 {
            max_mip_levels as usize
        } else {
            8
        };

        let mut ring_guard = self.ring.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire GlStagingRing lock".to_string())
        })?;
        let (_slot_idx, slot) = ring_guard.acquire_slot()?;

        unsafe {
            (gl.glPixelStorei)(GL_UNPACK_ALIGNMENT, 1);
            (gl.glPixelStorei)(GL_PACK_ALIGNMENT, 1);

            let upload_start = Instant::now();

            // Calculate intermediate downscale passes (successive halving)
            let mut pass_dims = Vec::new();
            let mut curr_w = src.width;
            let mut curr_h = src.height;

            while (curr_w > dst.width * 2 || curr_h > dst.height * 2) && pass_dims.len() < max_passes {
                curr_w = (curr_w / 2).max(dst.width);
                curr_h = (curr_h / 2).max(dst.height);
                pass_dims.push((curr_w, curr_h));
                if curr_w == dst.width && curr_h == dst.height {
                    break;
                }
            }
            pass_dims.push((dst.width, dst.height));

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

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.src_fbo);
            (gl.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, src_tex, 0);

            // Pre-allocate destination texture
            let dst_tex = slot.ensure_dst_texture(
                &self.ctx,
                src.width,
                src.height,
                dst_internal,
                dst_format,
                dst_type,
            )?;

            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, slot.dst_fbo);
            (gl.glFramebufferTexture2D)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst_tex, 0);
            let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

            let blit_start = Instant::now();
            let fbos = [slot.src_fbo, slot.dst_fbo];
            let textures = [src_tex, dst_tex];
            let mut read_idx = 0;
            let mut prev_w = src.width;
            let mut prev_h = src.height;

            for &(target_w, target_h) in &pass_dims {
                let write_idx = 1 - read_idx;

                (gl.glBindTexture)(GL_TEXTURE_2D, textures[write_idx]);
                (gl.glTexImage2D)(
                    GL_TEXTURE_2D, 0, dst_internal as i32,
                    target_w as i32, target_h as i32, 0,
                    dst_format, dst_type, std::ptr::null(),
                );

                (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, fbos[read_idx]);
                (gl.glBindFramebuffer)(GL_DRAW_FRAMEBUFFER, fbos[write_idx]);
                (gl.glBlitFramebuffer)(
                    0, 0, prev_w as i32, prev_h as i32,
                    0, 0, target_w as i32, target_h as i32,
                    GL_COLOR_BUFFER_BIT,
                    GL_LINEAR,
                );

                read_idx = write_idx;
                prev_w = target_w;
                prev_h = target_h;
            }

            slot.signal_fence(&self.ctx);
            let blit_ms = blit_start.elapsed().as_secs_f64() * 1000.0;

            // Download final output
            let download_start = Instant::now();
            (gl.glBindFramebuffer)(GL_READ_FRAMEBUFFER, fbos[read_idx]);
            (gl.glReadPixels)(
                0, 0, dst.width as i32, dst.height as i32,
                dst_format, dst_type, dst.data.as_mut_ptr() as *mut std::ffi::c_void,
            );
            (gl.glBindFramebuffer)(GL_FRAMEBUFFER, 0);
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
