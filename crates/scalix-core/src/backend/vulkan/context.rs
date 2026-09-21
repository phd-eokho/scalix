//! Headless Vulkan Context Initialization & Device Management

use std::sync::Arc;
use ash::{vk, Device, Entry, Instance};
use crate::types::{Result, ScalixError};

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub timestamp_period: f32,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl VulkanContext {
    /// Initializes a new headless Vulkan context.
    pub fn new() -> Result<Arc<Self>> {
        let entry = unsafe {
            Entry::load().map_err(|_e| {
                ScalixError::BackendUnavailable(crate::types::BackendType::Vulkan)
            })?
        };

        let app_name = std::ffi::CStr::from_bytes_with_nul(b"Scalix Engine\0")
            .map_err(|e| ScalixError::ExecutionFailed(format!("Invalid application name: {e}")))?;
        let engine_name = std::ffi::CStr::from_bytes_with_nul(b"Scalix\0")
            .map_err(|e| ScalixError::ExecutionFailed(format!("Invalid engine name: {e}")))?;

        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        let instance_create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        let instance = unsafe {
            entry.create_instance(&instance_create_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create Vulkan instance: {}", e))
            })?
        };

        // 1. Enumerate and pick the best physical device
        let physical_devices = unsafe {
            instance.enumerate_physical_devices().map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to enumerate physical devices: {}", e))
            })?
        };

        if physical_devices.is_empty() {
            return Err(ScalixError::BackendUnavailable(crate::types::BackendType::Vulkan));
        }

        // Pick discrete GPU if available, else first supported device
        let physical_device = physical_devices
            .iter()
            .copied()
            .max_by_key(|&pdev| {
                let props = unsafe { instance.get_physical_device_properties(pdev) };
                match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                    _ => 0,
                }
            })
            .ok_or_else(|| ScalixError::BackendUnavailable(crate::types::BackendType::Vulkan))?;

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // 2. Find a queue family supporting graphics/transfer/compute
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .position(|(_, info)| {
                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    || info.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .ok_or_else(|| {
                ScalixError::ExecutionFailed("No suitable Vulkan queue family found".to_string())
            })? as u32;

        // 3. Create logical device
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };

        let queue_infos = [queue_info];
        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: queue_infos.as_ptr(),
            ..Default::default()
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(|e| {
                    ScalixError::ExecutionFailed(format!("Failed to create logical device: {}", e))
                })?
        };

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // 4. Create Command Pool
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        let command_pool = unsafe {
            device.create_command_pool(&pool_info, None).map_err(|e| {
                ScalixError::ExecutionFailed(format!("Failed to create command pool: {}", e))
            })?
        };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let timestamp_period = device_properties.limits.timestamp_period;

        Ok(Arc::new(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            memory_properties,
            timestamp_period,
        }))
    }

    /// Finds a compatible memory type index matching the required memory type bits and properties.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        for (i, mem_type) in self.memory_properties.memory_types.iter().enumerate() {
            if (type_filter & (1 << i)) != 0 && mem_type.property_flags.contains(properties) {
                return Ok(i as u32);
            }
        }
        Err(ScalixError::ExecutionFailed(
            "Failed to find suitable memory type".to_string(),
        ))
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
