//! Offscreen Raster Graphics Pipeline (`vkCmdDraw` with hardware sampler)
//!
//! Utilizes GPU texture sampling hardware (bilinear filtering)
//! via fullscreen triangle rendering into offscreen framebuffers.

use std::sync::Arc;
use ash::vk;
use crate::backend::vulkan::blit::{to_vk_filter, to_vk_format_info};
use crate::backend::vulkan::context::VulkanContext;
use crate::backend::vulkan::util::{CommandBufferGuard, GpuBuffer, GpuImage, QueryPoolGuard};
use crate::backend::vulkan::{VulkanPipeline, VulkanStrategy};
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{FilterMode, ImageDesc, ImageDescMut, ResizeOptions, Result, ScalixError};

/// Pre-compiled SPIR-V binary bytecode for fullscreen triangle vertex shader.
///
/// ```glsl
/// #version 450
/// layout(location = 0) out vec2 outUV;
/// void main() {
///     float x = float((gl_VertexIndex << 1) & 2);
///     float y = float(gl_VertexIndex & 2);
///     outUV = vec2(x, y);
///     gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
/// }
/// ```
pub const VERT_SPV: &[u32] = &[
    119734787, 65536, 524289, 50, 0, 131089, 1, 196622, 0, 1, 524303, 0, 35, 1852399981, 0, 12,
    14, 16, 262215, 12, 11, 42, 262215, 14, 30, 0, 196679, 10, 2, 327752, 10, 0, 11, 0, 131091, 1,
    196641, 2, 1, 262165, 3, 32, 0, 262165, 4, 32, 1, 196630, 5, 32, 262167, 6, 5, 2, 262167, 7,
    5, 4, 262187, 3, 8, 6, 262172, 9, 6, 8, 196638, 10, 7, 262176, 11, 1, 3, 262203, 11, 12, 1,
    262176, 13, 3, 6, 262203, 13, 14, 3, 262176, 15, 3, 10, 262203, 15, 16, 3, 262176, 17, 3, 7,
    262176, 32, 7, 9, 262176, 37, 7, 6, 262187, 4, 18, 0, 262187, 5, 19, 3212836864, 262187, 5,
    20, 1065353216, 262187, 5, 21, 0, 327724, 6, 22, 19, 19, 327724, 6, 23, 20, 19, 327724, 6,
    24, 20, 20, 327724, 6, 25, 19, 20, 327724, 6, 26, 21, 21, 327724, 6, 27, 20, 21, 327724, 6,
    28, 21, 20, 589868, 9, 29, 22, 23, 24, 24, 25, 22, 589868, 9, 30, 26, 27, 24, 24, 28, 26,
    327734, 1, 35, 0, 2, 131320, 36, 262203, 32, 33, 7, 262203, 32, 34, 7, 196670, 33, 29,
    196670, 34, 30, 262205, 3, 38, 12, 327745, 37, 39, 33, 38, 262205, 6, 40, 39, 327745, 37,
    41, 34, 38, 262205, 6, 42, 41, 196670, 14, 42, 327761, 5, 43, 40, 0, 327761, 5, 44, 40, 1,
    458832, 7, 45, 43, 44, 21, 20, 327745, 17, 46, 16, 18, 196670, 46, 45, 65789, 65592,
];

/// Pre-compiled SPIR-V binary bytecode for texture sampling fragment shader.
///
/// ```glsl
/// #version 450
/// layout(binding = 0) uniform sampler2D uTex;
/// layout(location = 0) in vec2 inUV;
/// layout(location = 0) out vec4 outColor;
/// void main() {
///     outColor = texture(uTex, inUV);
/// }
/// ```
pub const FRAG_SPV: &[u32] = &[
    119734787, 65536, 524289, 20, 0, 131089, 1, 196622, 0, 1, 458767, 4, 15, 1852399981, 0, 12,
    14, 196624, 15, 7, 262215, 10, 34, 0, 262215, 10, 33, 0, 262215, 12, 30, 0, 262215, 14, 30,
    0, 131091, 2, 196641, 3, 2, 196630, 4, 32, 262167, 5, 4, 2, 262167, 6, 4, 4, 589849, 7, 4,
    1, 0, 0, 0, 1, 0, 196635, 8, 7, 262176, 9, 0, 8, 262203, 9, 10, 0, 262176, 11, 1, 5, 262203,
    11, 12, 1, 262176, 13, 3, 6, 262203, 13, 14, 3, 327734, 2, 15, 0, 3, 131320, 16, 262205, 8,
    17, 10, 262205, 5, 18, 12, 327767, 6, 19, 17, 18, 196670, 14, 19, 65789, 65592,
];

struct RasterPipelineResources {
    sampler: vk::Sampler,
    desc_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    framebuffer: vk::Framebuffer,
    desc_pool: vk::DescriptorPool,
    ctx: Arc<VulkanContext>,
}

impl Drop for RasterPipelineResources {
    fn drop(&mut self) {
        let d = &self.ctx.device;
        unsafe {
            if self.desc_pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.desc_pool, None);
            }
            if self.framebuffer != vk::Framebuffer::null() {
                d.destroy_framebuffer(self.framebuffer, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                d.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                d.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                d.destroy_render_pass(self.render_pass, None);
            }
            if self.desc_set_layout != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.desc_set_layout, None);
            }
            if self.vert_module != vk::ShaderModule::null() {
                d.destroy_shader_module(self.vert_module, None);
            }
            if self.frag_module != vk::ShaderModule::null() {
                d.destroy_shader_module(self.frag_module, None);
            }
            if self.sampler != vk::Sampler::null() {
                d.destroy_sampler(self.sampler, None);
            }
        }
    }
}

pub struct VulkanRasterResizer {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
}

impl VulkanRasterResizer {
    #[inline]
    #[must_use]
    pub fn new(ctx: Arc<VulkanContext>, profiler: Arc<dyn Profiler>) -> Self {
        Self { ctx, profiler }
    }

    pub fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, filter: FilterMode) -> Result<()> {
        let (vk_format, src_is_rgb) = to_vk_format_info(src.format)?;
        let (dst_vk_format, dst_is_rgb) = to_vk_format_info(dst.format)?;

        if vk_format != dst_vk_format || src_is_rgb != dst_is_rgb {
            return Err(ScalixError::ExecutionFailed(format!(
                "Raster format conversion not supported directly: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let is_profiling = self.profiler.is_enabled();
        let t0_wall = if is_profiling { Some(std::time::Instant::now()) } else { None };

        let vk_filter = to_vk_filter(filter);
        let device = &self.ctx.device;

        let src_size = if src_is_rgb {
            (src.width as usize * src.height as usize * 4) as vk::DeviceSize
        } else {
            src.data.len() as vk::DeviceSize
        };

        let dst_size = if dst_is_rgb {
            (dst.width as usize * dst.height as usize * 4) as vk::DeviceSize
        } else {
            dst.data.len() as vk::DeviceSize
        };

        unsafe {
            // 1. Create Staging Buffers via RAII guards
            let src_staging = GpuBuffer::allocate(
                &self.ctx,
                src_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let src_staging_buf = src_staging.buffer;
            let src_staging_mem = src_staging.memory;

            // Host Unpack RGB888 -> RGBA8888 if needed
            let t_unpack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let ptr = device
                .map_memory(src_staging_mem, 0, src_size, vk::MemoryMapFlags::empty())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to map src staging memory: {e}")))?
                as *mut u8;
            if src_is_rgb {
                let num_pixels = (src.width * src.height) as usize;
                for p in 0..num_pixels {
                    let s_idx = p * 3;
                    let d_idx = p * 4;
                    *ptr.add(d_idx) = src.data[s_idx];
                    *ptr.add(d_idx + 1) = src.data[s_idx + 1];
                    *ptr.add(d_idx + 2) = src.data[s_idx + 2];
                    *ptr.add(d_idx + 3) = 255;
                }
            } else {
                std::ptr::copy_nonoverlapping(src.data.as_ptr(), ptr, src.data.len());
            }
            device.unmap_memory(src_staging_mem);
            let host_unpack_ms = t_unpack_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let dst_staging = GpuBuffer::allocate(
                &self.ctx,
                dst_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let dst_staging_buf = dst_staging.buffer;
            let dst_staging_mem = dst_staging.memory;

            // 2. Create Source Image (SAMPLED | TRANSFER_DST) and Destination Image (COLOR_ATTACHMENT | TRANSFER_SRC)
            let mut src_gpu_img = GpuImage::allocate(
                &self.ctx,
                vk_format,
                src.width,
                src.height,
                1,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            )?;
            let src_image = src_gpu_img.image;

            let mut dst_gpu_img = GpuImage::allocate(
                &self.ctx,
                vk_format,
                dst.width,
                dst.height,
                1,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;
            let dst_image = dst_gpu_img.image;

            // 3. Create Image Views via GpuImage helper
            let src_view = src_gpu_img.create_view(vk_format, vk::ImageAspectFlags::COLOR)?;
            let dst_view = dst_gpu_img.create_view(vk_format, vk::ImageAspectFlags::COLOR)?;

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            // 4. Create Hardware Sampler
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk_filter,
                min_filter: vk_filter,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                anisotropy_enable: vk::FALSE,
                max_anisotropy: 1.0,
                border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
                unnormalized_coordinates: vk::FALSE,
                compare_enable: vk::FALSE,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                min_lod: 0.0,
                max_lod: 0.0,
                ..Default::default()
            };
            let sampler = device
                .create_sampler(&sampler_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create sampler: {e}")))?;

            // 5. Create Descriptor Set Layout, Pipeline Layout & Render Pass
            let dsl_binding = vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            };
            let dsl_info = vk::DescriptorSetLayoutCreateInfo {
                binding_count: 1,
                p_bindings: &dsl_binding,
                ..Default::default()
            };
            let desc_set_layout = device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create descriptor set layout: {e}")))?;

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &desc_set_layout,
                ..Default::default()
            };
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create pipeline layout: {e}")))?;

            let color_attachment = vk::AttachmentDescription {
                format: vk_format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            };
            let color_attachment_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            let subpass = vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: &color_attachment_ref,
                ..Default::default()
            };
            let render_pass_info = vk::RenderPassCreateInfo {
                attachment_count: 1,
                p_attachments: &color_attachment,
                subpass_count: 1,
                p_subpasses: &subpass,
                ..Default::default()
            };
            let render_pass = device
                .create_render_pass(&render_pass_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create render pass: {e}")))?;

            // 6. Create Shaders & Graphics Pipeline
            let vert_module_info = vk::ShaderModuleCreateInfo {
                code_size: std::mem::size_of_val(VERT_SPV),
                p_code: VERT_SPV.as_ptr(),
                ..Default::default()
            };
            let vert_module = device
                .create_shader_module(&vert_module_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create vert shader module: {e}")))?;

            let frag_module_info = vk::ShaderModuleCreateInfo {
                code_size: std::mem::size_of_val(FRAG_SPV),
                p_code: FRAG_SPV.as_ptr(),
                ..Default::default()
            };
            let frag_module = device
                .create_shader_module(&frag_module_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create frag shader module: {e}")))?;

            let main_name = std::ffi::CStr::from_bytes_with_nul(b"main\0")
                .map_err(|e| ScalixError::ExecutionFailed(format!("Invalid shader entry point CStr: {e}")))?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert_module,
                    p_name: main_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag_module,
                    p_name: main_name.as_ptr(),
                    ..Default::default()
                },
            ];

            let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                primitive_restart_enable: vk::FALSE,
                ..Default::default()
            };
            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };
            let rasterizer = vk::PipelineRasterizationStateCreateInfo {
                depth_clamp_enable: vk::FALSE,
                rasterizer_discard_enable: vk::FALSE,
                polygon_mode: vk::PolygonMode::FILL,
                line_width: 1.0,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                ..Default::default()
            };
            let multisampling = vk::PipelineMultisampleStateCreateInfo {
                sample_shading_enable: vk::FALSE,
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
                color_write_mask: vk::ColorComponentFlags::RGBA,
                blend_enable: vk::FALSE,
                ..Default::default()
            };
            let color_blending = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &color_blend_attachment,
                ..Default::default()
            };
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };

            let pipeline_info = vk::GraphicsPipelineCreateInfo {
                stage_count: 2,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input_info,
                p_input_assembly_state: &input_assembly,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterizer,
                p_multisample_state: &multisampling,
                p_color_blend_state: &color_blending,
                p_dynamic_state: &dynamic_state_info,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| ScalixError::ExecutionFailed(format!("Failed to create graphics pipeline: {:?}", e)))?[0];

            // 7. Create Framebuffer
            let fb_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count: 1,
                p_attachments: &dst_view,
                width: dst.width,
                height: dst.height,
                layers: 1,
                ..Default::default()
            };
            let framebuffer = device
                .create_framebuffer(&fb_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create framebuffer: {e}")))?;

            // 8. Create Descriptor Pool & Descriptor Set
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            };
            let desc_pool_info = vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: 1,
                p_pool_sizes: &pool_size,
                ..Default::default()
            };
            let desc_pool = device
                .create_descriptor_pool(&desc_pool_info, None)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to create descriptor pool: {e}")))?;

            let desc_alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool: desc_pool,
                descriptor_set_count: 1,
                p_set_layouts: &desc_set_layout,
                ..Default::default()
            };
            let desc_set = device
                .allocate_descriptor_sets(&desc_alloc_info)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to allocate descriptor sets: {e}")))?
                .into_iter()
                .next()
                .ok_or_else(|| ScalixError::ExecutionFailed("Allocated empty descriptor sets".to_string()))?;

            let desc_image_info = vk::DescriptorImageInfo {
                sampler,
                image_view: src_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write_desc = vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &desc_image_info,
                ..Default::default()
            };
            device.update_descriptor_sets(&[write_desc], &[]);

            // RAII guard to safely drop pipeline resources
            let _pipeline_res = RasterPipelineResources {
                sampler,
                desc_set_layout,
                pipeline_layout,
                render_pass,
                vert_module,
                frag_module,
                pipeline,
                framebuffer,
                desc_pool,
                ctx: Arc::clone(&self.ctx),
            };

            // 9. Command Buffer & Timestamps via RAII guards
            let cmd_guard = CommandBufferGuard::allocate(&self.ctx)?;
            let cmd_buf = cmd_guard.cmd_buf;

            let query_guard = if is_profiling {
                QueryPoolGuard::new(&self.ctx, 4)
            } else {
                None
            };
            let query_pool = query_guard.as_ref().map(|q| q.pool);

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device.begin_command_buffer(cmd_buf, &begin_info).unwrap();

            if let Some(qp) = query_pool {
                device.cmd_reset_query_pool(cmd_buf, qp, 0, 4);
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            }

            // Transition src_image UNDEFINED -> TRANSFER_DST_OPTIMAL
            let barrier_src_upload = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                image: src_image,
                subresource_range,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src_upload],
            );

            // Copy src staging to src_image
            let buffer_image_copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: src.width, height: src.height, depth: 1 },
            };
            device.cmd_copy_buffer_to_image(
                cmd_buf,
                src_staging_buf,
                src_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[buffer_image_copy],
            );

            // Transition src_image TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
            let barrier_src_sample = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                image: src_image,
                subresource_range,
                ..Default::default()
            };

            // Transition dst_image UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
            let barrier_dst_render = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                image: dst_image,
                subresource_range,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_src_sample, barrier_dst_render],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 1);
            }

            // Begin Render Pass & Draw Fullscreen Triangle
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: dst.width, height: dst.height },
            };
            let pass_begin_info = vk::RenderPassBeginInfo {
                render_pass,
                framebuffer,
                render_area,
                clear_value_count: 0,
                p_clear_values: std::ptr::null(),
                ..Default::default()
            };
            device.cmd_begin_render_pass(cmd_buf, &pass_begin_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            device.cmd_set_viewport(
                cmd_buf,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: dst.width as f32,
                    height: dst.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(cmd_buf, 0, &[render_area]);
            device.cmd_draw(cmd_buf, 6, 1, 0, 0);
            device.cmd_end_render_pass(cmd_buf);

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, qp, 2);
            }

            // Transition dst_image COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL for readback
            let barrier_dst_readback = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                image: dst_image,
                subresource_range,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_dst_readback],
            );

            // Copy dst_image to dst staging buffer
            let dst_buffer_image_copy = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: dst.width, height: dst.height, depth: 1 },
            };
            device.cmd_copy_image_to_buffer(
                cmd_buf,
                dst_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_staging_buf,
                &[dst_buffer_image_copy],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 3);
            }

            device
                .end_command_buffer(cmd_buf)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to end command buffer: {e}")))?;

            // Submit and synchronize
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            let t_sync_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            device
                .queue_submit(self.ctx.queue, &[submit_info], vk::Fence::null())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to submit queue: {e}")))?;
            device
                .queue_wait_idle(self.ctx.queue)
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to wait for queue idle: {e}")))?;
            let driver_sync_ms = t_sync_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            let mut gpu_upload_ms = 0.0;
            let mut gpu_pure_blit_ms = 0.0;
            let mut gpu_download_ms = 0.0;

            if let Some(qp) = query_pool {
                let mut timestamps = [0u64; 4];
                if device.get_query_pool_results(
                    qp,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                ).is_ok() {
                    let period_ms = (self.ctx.timestamp_period as f64) * 1e-6;
                    gpu_upload_ms = timestamps[1].saturating_sub(timestamps[0]) as f64 * period_ms;
                    gpu_pure_blit_ms = timestamps[2].saturating_sub(timestamps[1]) as f64 * period_ms;
                    gpu_download_ms = timestamps[3].saturating_sub(timestamps[2]) as f64 * period_ms;
                }
            }

            // Copy result from staging buffer to destination slice (with RGBA -> RGB repack if necessary)
            let t_repack_start = if is_profiling { Some(std::time::Instant::now()) } else { None };
            let out_ptr = device
                .map_memory(dst_staging_mem, 0, dst_size, vk::MemoryMapFlags::empty())
                .map_err(|e| ScalixError::ExecutionFailed(format!("Failed to map dst staging memory: {e}")))?
                as *const u8;
            if dst_is_rgb {
                let num_pixels = (dst.width * dst.height) as usize;
                for p in 0..num_pixels {
                    let s_idx = p * 4;
                    let d_idx = p * 3;
                    dst.data[d_idx] = *out_ptr.add(s_idx);
                    dst.data[d_idx + 1] = *out_ptr.add(s_idx + 1);
                    dst.data[d_idx + 2] = *out_ptr.add(s_idx + 2);
                }
            } else {
                std::ptr::copy_nonoverlapping(out_ptr, dst.data.as_mut_ptr(), dst.data.len());
            }
            device.unmap_memory(dst_staging_mem);
            let host_repack_ms = t_repack_start.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);

            if is_profiling {
                let total_wall_ms = t0_wall.map(|t| t.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
                self.profiler.record(ProfileMetrics {
                    host_unpack_ms,
                    gpu_upload_ms,
                    gpu_pure_blit_ms,
                    gpu_download_ms,
                    host_repack_ms,
                    driver_sync_ms,
                    total_wall_ms,
                });
            }

            Ok(())
        }
    }
}

impl VulkanPipeline for VulkanRasterResizer {
    #[inline]
    fn name(&self) -> &'static str {
        "VulkanRasterResizer"
    }

    #[inline]
    fn strategy(&self) -> VulkanStrategy {
        VulkanStrategy::Raster
    }

    #[inline]
    fn process(&self, src: &ImageDesc, dst: &mut ImageDescMut, options: &ResizeOptions) -> Result<()> {
        self.process(src, dst, options.filter)
    }
}
