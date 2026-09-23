//! Programmable Compute Shader Resampler Pipeline (`vkCmdDispatch`)
//!
//! Fused GPU Compute Kernels for direct packed RGB888 / BGR888 and 32-bit RGBA8888 / BGRA8888 resampling.
//! Avoids intermediate RGBA8888 texture allocations and multi-pass blits for packed formats, while
//! accelerating high-order spatial filters (Bicubic, Lanczos3) on headless offscreen GPUs.
//! Provides a pluggable kernel registry for dynamic custom compute shaders.

use crate::backend::vulkan::context::VulkanContext;
use crate::backend::vulkan::ring::{VulkanStagingRing, DEFAULT_RING_SLOTS};
use crate::backend::vulkan::{VulkanPipeline, VulkanStrategy};
use crate::profiler::{ProfileMetrics, Profiler};
use crate::types::{
    FilterMode, ImageDesc, ImageDescMut, PixelFormat, ResizeOptions, Result, ScalixError,
};
use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Pre-compiled SPIR-V binary bytecode for direct RGB888 nearest resize compute kernel.
pub const RGB888_RESIZE_NEAREST_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgb888_resize_nearest.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGB888 bilinear resize compute kernel.
pub const RGB888_RESIZE_BILINEAR_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgb888_resize_bilinear.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGB888 bicubic resize compute kernel.
///
/// NOTE (Reference Implementation):
/// Utilizes 2D separable Catmull-Rom cubic spline filtering ($a = -0.5$) over a $4 \times 4$ tap window.
/// Reference: Keys, R. (1981). "Cubic convolution interpolation for digital image processing",
/// IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(6), 1153-1160.
pub const RGB888_RESIZE_BICUBIC_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgb888_resize_bicubic.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGB888 Lanczos3 resize compute kernel.
///
/// NOTE (Reference Implementation):
/// Utilizes 2D separable 3-lobe sinc-windowed sinc filtering ($a = 3$) over a $6 \times 6$ tap window
/// with dynamic weight normalization.
/// Reference: Lanczos, C. (1956). "Applied Analysis", Prentice-Hall;
/// Turkowski, K. (1990). "Filters for Common Resampling Tasks", Graphics Gems.
pub const RGB888_RESIZE_LANCZOS3_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgb888_resize_lanczos3.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGBA8888 nearest resize compute kernel.
pub const RGBA8888_RESIZE_NEAREST_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgba8888_resize_nearest.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGBA8888 bilinear resize compute kernel.
pub const RGBA8888_RESIZE_BILINEAR_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgba8888_resize_bilinear.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGBA8888 bicubic resize compute kernel.
///
/// NOTE (Reference Implementation):
/// Utilizes 2D separable Catmull-Rom cubic spline filtering ($a = -0.5$) over a $4 \times 4$ tap window
/// across 4 color and alpha channels.
/// Reference: Keys, R. (1981). "Cubic convolution interpolation for digital image processing",
/// IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(6), 1153-1160.
pub const RGBA8888_RESIZE_BICUBIC_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgba8888_resize_bicubic.spv");

/// Pre-compiled SPIR-V binary bytecode for direct RGBA8888 Lanczos3 resize compute kernel.
///
/// NOTE (Reference Implementation):
/// Utilizes 2D separable 3-lobe sinc-windowed sinc filtering ($a = 3$) over a $6 \times 6$ tap window
/// with dynamic weight normalization across RGBA channels.
/// Reference: Lanczos, C. (1956). "Applied Analysis", Prentice-Hall;
/// Turkowski, K. (1990). "Filters for Common Resampling Tasks", Graphics Gems.
pub const RGBA8888_RESIZE_LANCZOS3_COMP_SPV: &[u8] =
    include_bytes!("shaders/rgba8888_resize_lanczos3.spv");

/// Default local workgroup size along X dimension.
pub const DEFAULT_WORKGROUP_SIZE_X: u32 = 64;

/// Default packed pixels processed per compute thread.
pub const DEFAULT_PIXELS_PER_THREAD: u32 = 4;

/// Trailing padding bytes to prevent word-boundary read/write overruns in compute buffers.
pub const BUFFER_TAIL_PADDING_BYTES: usize = 16;

/// Number of timestamp queries for execution profiling (start and end).
pub const TIMESTAMP_QUERY_COUNT: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ComputePushConsts {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub scale_x: f32,
    pub scale_y: f32,
}

/// Unique lookup key identifying a compute kernel by pixel format and filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeKernelKey {
    pub format: PixelFormat,
    pub filter: FilterMode,
}

impl ComputeKernelKey {
    #[inline]
    pub const fn new(format: PixelFormat, filter: FilterMode) -> Self {
        Self { format, filter }
    }
}

/// Scoped RAII guard guaranteeing deterministic destruction of transient VkDescriptorPool.
struct DescriptorPoolGuard<'a> {
    device: &'a ash::Device,
    pool: vk::DescriptorPool,
}

impl<'a> DescriptorPoolGuard<'a> {
    #[inline]
    fn new(device: &'a ash::Device, pool: vk::DescriptorPool) -> Self {
        Self { device, pool }
    }

    #[inline]
    fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl<'a> Drop for DescriptorPoolGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Represents an individual compiled compute shader kernel and its GPU pipeline.
pub struct ComputeKernel {
    device: ash::Device,
    pub pipeline: vk::Pipeline,
    pub shader_module: vk::ShaderModule,
    pub workgroup_size_x: u32,
    pub pixels_per_thread: u32,
}

impl Drop for ComputeKernel {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
        }
    }
}

/// Fused GPU compute resizer executing direct 24-bit RGB888 / BGR888 and 32-bit RGBA8888 / BGRA8888 resampling kernels.
/// Supports pluggable compute shaders for custom spatial filters and image processing.
pub struct VulkanComputeResizer {
    ctx: Arc<VulkanContext>,
    profiler: Arc<dyn Profiler>,
    desc_layout: vk::DescriptorSetLayout,
    pipe_layout: vk::PipelineLayout,
    kernels: Mutex<HashMap<ComputeKernelKey, Arc<ComputeKernel>>>,
    ring: Mutex<VulkanStagingRing>,
}

impl VulkanComputeResizer {
    /// Initializes compute resizer pipelines and registers default nearest, bilinear, bicubic & lanczos3 kernels.
    pub fn new(ctx: Arc<VulkanContext>, profiler: Arc<dyn Profiler>) -> Result<Self> {
        let device = &ctx.device;

        // 1. Descriptor Set Layout: Binding 0 = Storage Buffer (src), Binding 1 = Storage Buffer (dst)
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                ..Default::default()
            },
        ];

        let dsl_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let desc_layout = unsafe {
            device
                .create_descriptor_set_layout(&dsl_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to create compute resizer descriptor layout: {e}"
                    ))
                })?
        };

        // 2. Pipeline Layout with push constants
        let push_const_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<ComputePushConsts>() as u32,
        };

        let pl_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &desc_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_const_range,
            ..Default::default()
        };
        let pipe_layout = unsafe {
            device.create_pipeline_layout(&pl_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!(
                    "Failed to create compute resizer pipeline layout: {e}"
                ))
            })?
        };

        let ring = Mutex::new(
            VulkanStagingRing::new(Arc::clone(&ctx), DEFAULT_RING_SLOTS).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create staging ring: {e}"))
            })?,
        );

        let resizer = Self {
            ctx,
            profiler,
            desc_layout,
            pipe_layout,
            kernels: Mutex::new(HashMap::new()),
            ring,
        };

        // Register default RGB888 / BGR888 kernels
        for &fmt in &[PixelFormat::Rgb888, PixelFormat::Bgr888] {
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Nearest,
                RGB888_RESIZE_NEAREST_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Bilinear,
                RGB888_RESIZE_BILINEAR_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Bicubic,
                RGB888_RESIZE_BICUBIC_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Lanczos3,
                RGB888_RESIZE_LANCZOS3_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
        }

        // Register default RGBA8888 / BGRA8888 kernels
        for &fmt in &[PixelFormat::Rgba8888, PixelFormat::Bgra8888] {
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Nearest,
                RGBA8888_RESIZE_NEAREST_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Bilinear,
                RGBA8888_RESIZE_BILINEAR_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Bicubic,
                RGBA8888_RESIZE_BICUBIC_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
            resizer.register_format_shader_with_layout(
                fmt,
                FilterMode::Lanczos3,
                RGBA8888_RESIZE_LANCZOS3_COMP_SPV,
                DEFAULT_WORKGROUP_SIZE_X,
                DEFAULT_PIXELS_PER_THREAD,
            )?;
        }

        Ok(resizer)
    }

    /// Compiles and registers a custom SPIR-V compute shader for RGB888/BGR888 for a given filter mode.
    #[inline]
    pub fn register_shader(&self, filter: FilterMode, spv_bytes: &[u8]) -> Result<()> {
        self.register_shader_with_layout(
            filter,
            spv_bytes,
            DEFAULT_WORKGROUP_SIZE_X,
            DEFAULT_PIXELS_PER_THREAD,
        )
    }

    /// Compiles and registers a custom compute shader for RGB888/BGR888 specifying workgroup and pixel layout.
    pub fn register_shader_with_layout(
        &self,
        filter: FilterMode,
        spv_bytes: &[u8],
        workgroup_size_x: u32,
        pixels_per_thread: u32,
    ) -> Result<()> {
        self.register_format_shader_with_layout(
            PixelFormat::Rgb888,
            filter,
            spv_bytes,
            workgroup_size_x,
            pixels_per_thread,
        )?;
        self.register_format_shader_with_layout(
            PixelFormat::Bgr888,
            filter,
            spv_bytes,
            workgroup_size_x,
            pixels_per_thread,
        )
    }

    /// Compiles and registers a custom compute shader for a specific pixel format.
    #[inline]
    pub fn register_format_shader(
        &self,
        format: PixelFormat,
        filter: FilterMode,
        spv_bytes: &[u8],
    ) -> Result<()> {
        self.register_format_shader_with_layout(
            format,
            filter,
            spv_bytes,
            DEFAULT_WORKGROUP_SIZE_X,
            DEFAULT_PIXELS_PER_THREAD,
        )
    }

    /// Compiles and registers a custom compute shader specifying format, workgroup and pixel layout.
    pub fn register_format_shader_with_layout(
        &self,
        format: PixelFormat,
        filter: FilterMode,
        spv_bytes: &[u8],
        workgroup_size_x: u32,
        pixels_per_thread: u32,
    ) -> Result<()> {
        let kernel = self.create_kernel(spv_bytes, workgroup_size_x, pixels_per_thread)?;
        let mut kernels = self.kernels.lock().map_err(|_| {
            ScalixError::ExecutionFailed("Failed to acquire compute kernel lock".to_string())
        })?;
        kernels.insert(ComputeKernelKey::new(format, filter), kernel);
        Ok(())
    }

    /// Creates and compiles a `ComputeKernel` instance from SPIR-V bytecode.
    pub fn create_kernel(
        &self,
        spv_bytes: &[u8],
        workgroup_size_x: u32,
        pixels_per_thread: u32,
    ) -> Result<Arc<ComputeKernel>> {
        let device = &self.ctx.device;

        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes)).map_err(|e| {
            ScalixError::ExecutionFailed(format!("Failed to parse SPIR-V byte stream: {e}"))
        })?;

        let sm_info = vk::ShaderModuleCreateInfo {
            code_size: words.len() * std::mem::size_of::<u32>(),
            p_code: words.as_ptr(),
            ..Default::default()
        };
        let shader_module = unsafe {
            device.create_shader_module(&sm_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!(
                    "Failed to create custom compute shader module: {e}"
                ))
            })?
        };

        let main_name = c"main";
        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: main_name.as_ptr(),
            ..Default::default()
        };
        let cp_info = vk::ComputePipelineCreateInfo {
            stage,
            layout: self.pipe_layout,
            ..Default::default()
        };
        let pipeline = unsafe {
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[cp_info], None) {
                Ok(pipes) => pipes[0],
                Err((_, e)) => {
                    device.destroy_shader_module(shader_module, None);
                    return Err(ScalixError::ExecutionFailed(format!(
                        "Failed to create custom compute pipeline: {e:?}"
                    )));
                }
            }
        };

        Ok(Arc::new(ComputeKernel {
            device: device.clone(),
            pipeline,
            shader_module,
            workgroup_size_x,
            pixels_per_thread,
        }))
    }

    /// Checks if a compute kernel is registered for RGB888 and the specified filter mode.
    #[inline]
    #[must_use]
    pub fn has_shader(&self, filter: FilterMode) -> bool {
        self.has_format_shader(PixelFormat::Rgb888, filter)
    }

    /// Checks if a compute kernel is registered for the specified format and filter mode.
    #[inline]
    #[must_use]
    pub fn has_format_shader(&self, format: PixelFormat, filter: FilterMode) -> bool {
        self.kernels
            .lock()
            .map(|k| k.contains_key(&ComputeKernelKey::new(format, filter)))
            .unwrap_or(false)
    }

    /// Resizes an image directly via GPU compute dispatch using pluggable registered kernels.
    pub fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        filter: FilterMode,
    ) -> Result<()> {
        if src.width == 0 || src.height == 0 || dst.width == 0 || dst.height == 0 {
            return Err(ScalixError::InvalidDimensions {
                width: if src.width == 0 { src.width } else { dst.width },
                height: if src.height == 0 {
                    src.height
                } else {
                    dst.height
                },
            });
        }

        if src.format != dst.format {
            return Err(ScalixError::ExecutionFailed(format!(
                "Compute direct resizer expects identical src and dst formats: {:?} -> {:?}",
                src.format, dst.format
            )));
        }

        let bpp = match src.format.bytes_per_pixel() {
            Some(bytes)
                if matches!(
                    src.format,
                    PixelFormat::Rgb888
                        | PixelFormat::Bgr888
                        | PixelFormat::Rgba8888
                        | PixelFormat::Bgra8888
                ) =>
            {
                bytes
            }
            _ => return Err(ScalixError::UnsupportedFormat(src.format)),
        };

        let src_min_stride = (src.width as usize).saturating_mul(bpp);
        let dst_min_stride = (dst.width as usize).saturating_mul(bpp);
        if src.stride < src_min_stride || dst.stride < dst_min_stride {
            return Err(ScalixError::InvalidStride {
                stride: if src.stride < src_min_stride {
                    src.stride
                } else {
                    dst.stride
                },
                min_stride: if src.stride < src_min_stride {
                    src_min_stride
                } else {
                    dst_min_stride
                },
            });
        }

        let kernel = {
            let kernels = self.kernels.lock().map_err(|_| {
                ScalixError::ExecutionFailed("Failed to acquire compute kernel lock".to_string())
            })?;
            let key = ComputeKernelKey::new(src.format, filter);
            kernels
                .get(&key)
                .cloned()
                .or_else(|| {
                    // Fallback to Bilinear for high-order filters if not explicitly registered
                    if matches!(
                        filter,
                        FilterMode::Bicubic | FilterMode::Lanczos3 | FilterMode::Area
                    ) {
                        kernels
                            .get(&ComputeKernelKey::new(src.format, FilterMode::Bilinear))
                            .cloned()
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    ScalixError::ExecutionFailed(format!(
                        "No compute shader kernel registered for format {:?} and filter mode: {:?}",
                        src.format, filter
                    ))
                })?
        };

        let is_profiling = self.profiler.is_enabled();
        let t0_wall = if is_profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let device = &self.ctx.device;
        let src_size = src.data.len().saturating_add(BUFFER_TAIL_PADDING_BYTES) as vk::DeviceSize;
        let dst_size = dst.data.len().saturating_add(BUFFER_TAIL_PADDING_BYTES) as vk::DeviceSize;

        unsafe {
            let mut ring_guard = self.ring.lock().map_err(|_| {
                ScalixError::ExecutionFailed("Failed to acquire VulkanStagingRing lock".to_string())
            })?;
            let (_slot_idx, slot) = ring_guard.acquire_slot()?;

            let src_staging = slot.ensure_src_staging(
                &self.ctx,
                src_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let src_buf = src_staging.buffer;
            let src_mem = src_staging.memory;

            let ptr = device
                .map_memory(src_mem, 0, src_size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to map src memory: {e}"))
                })? as *mut u8;
            std::ptr::copy_nonoverlapping(src.data.as_ptr(), ptr, src.data.len());
            std::ptr::write_bytes(ptr.add(src.data.len()), 0, BUFFER_TAIL_PADDING_BYTES);
            device.unmap_memory(src_mem);

            let dst_staging = slot.ensure_dst_staging(
                &self.ctx,
                dst_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            )?;
            let dst_buf = dst_staging.buffer;
            let dst_mem = dst_staging.memory;

            // Transient descriptor pool for binding 0 & 1 managed via RAII guard
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            }];
            let pool_info = vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };
            let desc_pool = device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to create compute desc pool: {e}"))
                })?;
            let desc_pool_guard = DescriptorPoolGuard::new(device, desc_pool);

            let set_layouts = [self.desc_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool: desc_pool_guard.pool(),
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                ..Default::default()
            };
            let desc_set = device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to allocate compute desc set: {e}"))
            })?[0];

            let src_buf_info = [vk::DescriptorBufferInfo {
                buffer: src_buf,
                offset: 0,
                range: src_size,
            }];
            let dst_buf_info = [vk::DescriptorBufferInfo {
                buffer: dst_buf,
                offset: 0,
                range: dst_size,
            }];
            let descriptor_writes = [
                vk::WriteDescriptorSet {
                    dst_set: desc_set,
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: src_buf_info.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: desc_set,
                    dst_binding: 1,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: dst_buf_info.as_ptr(),
                    ..Default::default()
                },
            ];
            device.update_descriptor_sets(&descriptor_writes, &[]);

            let cmd_buf = slot.cmd_buf;
            let query_pool = if is_profiling { slot.query_pool } else { None };

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to begin compute cmd buf: {e}"))
                })?;

            if let Some(qp) = query_pool {
                device.cmd_reset_query_pool(cmd_buf, qp, 0, TIMESTAMP_QUERY_COUNT);
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
            }

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipe_layout,
                0,
                &[desc_set],
                &[],
            );

            let push_consts = ComputePushConsts {
                src_w: src.width,
                src_h: src.height,
                dst_w: dst.width,
                dst_h: dst.height,
                scale_x: src.width as f32 / dst.width as f32,
                scale_y: src.height as f32 / dst.height as f32,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &push_consts as *const _ as *const u8,
                std::mem::size_of::<ComputePushConsts>(),
            );
            device.cmd_push_constants(
                cmd_buf,
                self.pipe_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            let total_dst_pixels = dst.width.saturating_mul(dst.height);
            let pixels_per_workgroup = kernel
                .workgroup_size_x
                .saturating_mul(kernel.pixels_per_thread);
            if pixels_per_workgroup == 0 {
                return Err(ScalixError::ExecutionFailed(
                    "Compute kernel workgroup layout must be non-zero".to_string(),
                ));
            }
            let workgroups = total_dst_pixels.div_ceil(pixels_per_workgroup);
            device.cmd_dispatch(cmd_buf, workgroups.max(1), 1, 1);

            let memory_barrier = vk::MemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::HOST_READ,
                ..Default::default()
            };
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );

            if let Some(qp) = query_pool {
                device.cmd_write_timestamp(cmd_buf, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1);
            }

            device.end_command_buffer(cmd_buf).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to end compute cmd buf: {e}"))
            })?;

            let t_sync_start = if is_profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf,
                ..Default::default()
            };
            device
                .queue_submit(self.ctx.queue, &[submit_info], slot.fence)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Compute queue submit failed: {e}"))
                })?;

            slot.in_flight = true;
            device
                .wait_for_fences(&[slot.fence], true, u64::MAX)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!(
                        "Failed to wait for compute slot fence: {e}"
                    ))
                })?;
            device.reset_fences(&[slot.fence]).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to reset compute slot fence: {e}"))
            })?;
            slot.in_flight = false;

            let driver_sync_ms = t_sync_start
                .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);

            drop(desc_pool_guard);

            let mut gpu_pure_compute_ms = 0.0;
            if let Some(qp) = query_pool {
                let mut timestamps = [0u64; TIMESTAMP_QUERY_COUNT as usize];
                if device
                    .get_query_pool_results(
                        qp,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .is_ok()
                {
                    let period_ms = (self.ctx.timestamp_period as f64) * 1e-6;
                    gpu_pure_compute_ms =
                        timestamps[1].saturating_sub(timestamps[0]) as f64 * period_ms;
                }
            }

            let out_ptr = device
                .map_memory(dst_mem, 0, dst_size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to map dst memory: {e}"))
                })? as *const u8;
            std::ptr::copy_nonoverlapping(out_ptr, dst.data.as_mut_ptr(), dst.data.len());
            device.unmap_memory(dst_mem);

            if is_profiling {
                let total_wall_ms = t0_wall
                    .map(|t| t.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);

                self.profiler.record(ProfileMetrics {
                    host_unpack_ms: 0.0,
                    gpu_upload_ms: 0.0,
                    gpu_pure_blit_ms: gpu_pure_compute_ms,
                    gpu_download_ms: 0.0,
                    host_repack_ms: 0.0,
                    driver_sync_ms,
                    total_wall_ms,
                });
            }
        }

        Ok(())
    }
}

impl VulkanPipeline for VulkanComputeResizer {
    #[inline]
    fn name(&self) -> &'static str {
        "VulkanComputeResizer"
    }

    #[inline]
    fn strategy(&self) -> VulkanStrategy {
        VulkanStrategy::Compute
    }

    #[inline]
    fn process(
        &self,
        src: &ImageDesc,
        dst: &mut ImageDescMut,
        options: &ResizeOptions,
    ) -> Result<()> {
        self.process(src, dst, options.filter)
    }
}

impl Drop for VulkanComputeResizer {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        unsafe {
            if let Ok(mut kernels) = self.kernels.lock() {
                kernels.clear();
            }
            device.destroy_pipeline_layout(self.pipe_layout, None);
            device.destroy_descriptor_set_layout(self.desc_layout, None);
        }
    }
}
