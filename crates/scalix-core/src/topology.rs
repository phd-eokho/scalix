//! Hardware GPU Topology Probing & Architecture Classifier
//!
//! Provides unified multi-tier probing for discrete GPUs (PCIe VRAM) vs integrated GPUs / SoCs (UMA).
//! - Tier 1: Vulkan `VkPhysicalDeviceProperties` (Fast path)
//! - Tier 2: Linux DRM sysfs & PCI bus prober (Fallback when Vulkan is disabled/unavailable)
//! - Tier 3: Linux DMA-Heap & Android AHB prober (Platform UMA detection)

use crate::types::{GpuDeviceKind, GpuTopologyInfo};
use ash::vk;

/// Well-known PCI Vendor IDs for hardware classification.
pub mod pci_vendors {
    pub const NVIDIA: u32 = 0x10de;
    pub const AMD: u32 = 0x1002;
    pub const INTEL: u32 = 0x8086;
    pub const ARM: u32 = 0x13b5;
    pub const QUALCOMM: u32 = 0x5143;
    pub const BROADCOM: u32 = 0x14e4;
    pub const APPLE: u32 = 0x106b;
}

pub struct GpuTopology;

impl GpuTopology {
    /// Discovers the host GPU topology using the highest-priority discovery mechanism available.
    ///
    /// Automatically probes Vulkan first; if Vulkan is not available or fails to initialize,
    /// seamlessly falls back to Linux DRM sysfs inspection, PCI bus queries, or DMA-Heap detection.
    #[must_use]
    pub fn probe() -> GpuTopologyInfo {
        // 1. Attempt Vulkan headless probing
        if let Some(info) = Self::probe_vulkan_headless() {
            return info;
        }

        // 2. Fallback to Linux DRM sysfs & PCI prober
        #[cfg(target_os = "linux")]
        {
            let drm_info = Self::probe_drm_sysfs();
            if drm_info.device_kind != GpuDeviceKind::Unknown {
                return drm_info;
            }
        }

        // 3. Fallback to DMA-Heap / Platform SoC prober
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            if crate::dma::linux_dma_heap::LinuxDmaHeapAllocator::is_available() {
                return GpuTopologyInfo {
                    device_kind: GpuDeviceKind::Integrated,
                    device_name: "Generic Linux SoC (DMA-Heap UMA)".to_string(),
                    driver_name: "dma_heap".to_string(),
                    pci_vendor_id: None,
                    pci_device_id: None,
                };
            }
        }

        #[cfg(all(
            target_os = "android",
            any(target_arch = "aarch64", target_arch = "arm")
        ))]
        {
            if crate::dma::android_ahb::AndroidAhbAllocator::is_available() {
                return GpuTopologyInfo {
                    device_kind: GpuDeviceKind::Integrated,
                    device_name: "Android SoC (AHardwareBuffer UMA)".to_string(),
                    driver_name: "android_ahb".to_string(),
                    pci_vendor_id: None,
                    pci_device_id: None,
                };
            }
        }

        // 4. Default unknown/host fallback
        GpuTopologyInfo::default()
    }

    /// Attempts to probe GPU topology by instantiating a temporary headless Vulkan entry/instance.
    pub fn probe_vulkan_headless() -> Option<GpuTopologyInfo> {
        let entry = unsafe { ash::Entry::load().ok()? };
        let app_name = c"Scalix Topology Prober";
        let engine_name = c"Scalix";
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
        let instance = unsafe { entry.create_instance(&instance_create_info, None).ok()? };

        let pdevs = unsafe { instance.enumerate_physical_devices().ok()? };
        if pdevs.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return None;
        }

        // Select the highest priority device (Discrete > Integrated > Virtual)
        let selected_pdev = pdevs
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
            .unwrap_or(pdevs[0]);

        let info = Self::probe_from_vulkan(&instance, selected_pdev);
        unsafe { instance.destroy_instance(None) };
        Some(info)
    }

    /// Probes GPU topology from an existing Vulkan instance and physical device.
    pub fn probe_from_vulkan(
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
    ) -> GpuTopologyInfo {
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let device_kind = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => GpuDeviceKind::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => GpuDeviceKind::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU | vk::PhysicalDeviceType::CPU => {
                GpuDeviceKind::Virtual
            }
            _ => GpuDeviceKind::Unknown,
        };

        let device_name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        log::info!(
            "Discovered Vulkan GPU: '{}' (kind: {:?}, vendor: 0x{:04x}, dev: 0x{:04x}, UMA: {})",
            device_name,
            device_kind,
            props.vendor_id,
            props.device_id,
            device_kind == GpuDeviceKind::Integrated
        );

        GpuTopologyInfo {
            device_kind,
            device_name,
            driver_name: "vulkan".to_string(),
            pci_vendor_id: Some(props.vendor_id),
            pci_device_id: Some(props.device_id),
        }
    }

    /// Probes GPU topology by inspecting Linux DRM render nodes (`/sys/class/drm/renderD*`) and PCI subsystem.
    #[cfg(target_os = "linux")]
    pub fn probe_drm_sysfs() -> GpuTopologyInfo {
        for &node_path in crate::dma::linux_drm::DRM_CANDIDATE_PATHS {
            let file_name = match std::path::Path::new(node_path).file_name() {
                Some(n) => n.to_string_lossy().into_owned(),
                None => continue,
            };

            let sysfs_dev =
                std::path::PathBuf::from(format!("/sys/class/drm/{}/device", file_name));
            if !sysfs_dev.exists() {
                continue;
            }

            let subsystem = std::fs::read_link(sysfs_dev.join("subsystem"))
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();

            let driver = std::fs::read_link(sysfs_dev.join("driver"))
                .ok()
                .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
                .unwrap_or_default();

            let vendor_id = std::fs::read_to_string(sysfs_dev.join("vendor"))
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());

            let device_id = std::fs::read_to_string(sysfs_dev.join("device"))
                .ok()
                .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());

            let is_platform = subsystem.contains("platform");
            let is_pci = subsystem.contains("pci");

            let device_kind = match (is_platform, is_pci, vendor_id, driver.as_str()) {
                (true, _, _, _) => GpuDeviceKind::Integrated,
                (_, true, Some(pci_vendors::NVIDIA | pci_vendors::AMD), _) => {
                    GpuDeviceKind::Discrete
                }
                (
                    _,
                    true,
                    Some(
                        pci_vendors::INTEL
                        | pci_vendors::ARM
                        | pci_vendors::QUALCOMM
                        | pci_vendors::APPLE,
                    ),
                    _,
                ) => GpuDeviceKind::Integrated,
                (_, true, _, "nouveau" | "nvidia" | "amdgpu" | "radeon") => GpuDeviceKind::Discrete,
                _ => GpuDeviceKind::Unknown,
            };

            let device_name = format!("{file_name} ({driver})");

            log::info!(
                "Discovered DRM Sysfs GPU: '{}' (kind: {:?}, vendor: {:?}, dev: {:?}, subsystem: {}, UMA: {})",
                device_name,
                device_kind,
                vendor_id,
                device_id,
                subsystem,
                device_kind == GpuDeviceKind::Integrated
            );

            return GpuTopologyInfo {
                device_kind,
                device_name,
                driver_name: driver,
                pci_vendor_id: vendor_id,
                pci_device_id: device_id,
            };
        }

        GpuTopologyInfo::default()
    }
}
