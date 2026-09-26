# Android Buffer Architecture & Vendor-Side DMA-BUF Unification Guide

This document provides a comprehensive technical reference on Android's buffer management mechanisms, the evolution from legacy **ION** to **DMA-BUF Heaps**, the role of **`AHardwareBuffer`**, and the architectural blueprint for standardizing on **`dma-buf`** for vendor-side HAL and kernel implementations.

---

## 1. Evolution Overview & State of Technology (API 33+ / Android 14+ Focus)

For modern Android (API 33+ / Android 14+), legacy **ION is obsolete and eliminated from GKI**, and **HIDL Gralloc 4.0 is superseded by Stable AIDL Graphics Allocator & Mapper HALs**.

```
+-----------------------------------------------------------------------------------------------------+
| Era / Target OS         | Kernel Allocator        | Framework Graphics HAL    | Public NDK API       |
+-------------------------+-------------------------+---------------------------+----------------------+
| Android 11-13 (API 30-33)| DMA-BUF Heaps (/dev/dma_heap/*) | HIDL Gralloc 4.0 (hwbinder)| AHardwareBuffer      |
| Android 14+ (API 34+)   | DMA-BUF Heaps (/dev/dma_heap/*) | Stable AIDL Mapper/Allocator| AHardwareBuffer    |
+-----------------------------------------------------------------------------------------------------+
```

### Key Clarifications for Android 14+:
* **No ION Consideration:** All memory allocation relies solely on upstream Linux DMA-BUF Heaps (`/dev/dma_heap/*`) via `libdmabufheap`.
* **No Gralloc / AIDL Mapper in Vendor Pipelines:** Standalone vendor libraries, HALs (Camera, VPU, NPU, Display), and native daemons operate **exclusively with raw `dma-buf` file descriptors (`int fd`)**. They do not need to call or link against HIDL Gralloc or AIDL Mapper.
* **Framework Interoperability via Stable AIDL:** In Android 14+, the OS framework runtime (`libui.so`) uses Stable AIDL (`android.hardware.graphics.allocator`) instead of HIDL. NDK clients simply call `<android/hardware_buffer.h>` without worrying about underlying HAL implementations.

---

## 2. Multi-Layer Android Memory Architecture

```
===============================================================================
 USER SPACE (NDK / Application Layer)
  - AHardwareBuffer / AImageReader / ANativeWindow / Vulkan / OpenGLES
  - Purpose: High-level cross-API graphics and media buffer sharing.
===============================================================================
                      | (wraps)
                      v
===============================================================================
 ANDROID FRAMEWORK & HAL INTERMEDIARY
  - GraphicBuffer (C++ internal) / buffer_handle_t (native_handle_t)
  - Gralloc 4.0 / AIDL Allocator & Mapper HALs
  - SurfaceFlinger (Compositor) / Camera Framework
===============================================================================
                      | (translates / imports)
                      v
===============================================================================
 VENDOR DOMAIN (HALs, Daemons & Native Services)
  - Camera HAL (V4L2)  |  Display HAL (DRM/KMS)  |  GPU (EGL/Vulkan)
  - NPU / DSP Driver   |  Video Codec (VPU/V4L2)  |  Custom Accelerators
  - Standard Currency: Raw DMA-BUF File Descriptors (int fd)
===============================================================================
                      | (allocates / maps)
                      v
===============================================================================
 LINUX KERNEL (GKI / Drivers)
  - /dev/dma_heap/system, /dev/dma_heap/system-uncached, /dev/dma_heap/cma
  - DMA-BUF Subsystem (dma_buf_attach, dma_buf_map_attachment, dma_fence)
  - Sync File / Explicit Sync FDs
===============================================================================
```

---

## 3. Vendor-Side Architecture: Unifying on `dma-buf`

For vendor-side implementations (HALs, kernel drivers, background daemons, and hardware accelerators), **unifying around `dma-buf` file descriptors is the recommended standard**.

### Why `dma-buf` Over `AHardwareBuffer` on the Vendor Side
1. **Kernel & Driver Native:** Direct interoperability with upstream Linux subsystems (`V4L2`, `DRM/KMS`, `dma_buf_ops`).
2. **No NDK Dependency:** Operates independently of Android NDK / platform runtime constraints and lifecycles.
3. **True Zero-Copy Across IP Blocks:** Hardware blocks (ISP, GPU, NPU, Display) consume standard DMA-BUF handles directly.
4. **Standard IPC:** File descriptors transfer easily across processes via Unix domain sockets and AIDL/Binder (`ParcelFileDescriptor` / `ScopedFileDescriptor`).

---

## 4. Vendor Implementation Blueprint

### A. Memory Allocation with `libdmabufheap`
Modern AOSP provides `libdmabufheap` (`<BufferAllocator/BufferAllocator.h>`) for vendor code to allocate buffers from DMA-BUF heaps:

```cpp
#include <BufferAllocator/BufferAllocator.h>
#include <unistd.h>

BufferAllocator allocator;

// Allocate from system heap (cached) or CMA/custom vendor heaps
int dma_buf_fd = allocator.Alloc("system", buffer_size);
if (dma_buf_fd < 0) {
    // Handle allocation failure
}

// Pass dma_buf_fd to hardware drivers / HALs...

// Release memory when done
close(dma_buf_fd);
```

### B. Hardware Subsystem Integration

#### 1. Camera / ISP (V4L2)
* **Import to V4L2:** Set buffer type to `V4L2_MEMORY_DMABUF` and enqueue buffer via `VIDIOC_QBUF` passing `dma_buf_fd`.
* **Export from V4L2:** Allocate via `V4L2_MEMORY_MMAP` and export using `VIDIOC_EXPBUF` to obtain a `dma_buf_fd`.

#### 2. Display / Compositor (DRM/KMS)
* **Import to DRM:** Convert FD to DRM gem handle:
  ```c
  struct drm_prime_handle prime_req = { .fd = dma_buf_fd, .flags = 0 };
  ioctl(drm_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_req);
  uint32_t gem_handle = prime_req.handle;
  ```
* **Add Framebuffer:** Bind gem handle to DRM framebuffer using `drmModeAddFB2()`.

#### 3. GPU (Vulkan & OpenGL ES)
* **Vulkan:** Use the `VK_EXT_external_memory_dma_buf` extension:
  * Import via `VkImportMemoryFdInfoKHR` with `handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`.
* **OpenGL ES / EGL:** Use `EGL_EXT_image_dma_buf_import`:
  * Create `EGLImage` passing `EGL_LINUX_DMA_BUF_EXT` attributes (FD, width, height, stride, fourcc format).

#### 4. NPU / DSP / Custom Accelerators (Kernel Drivers)
* In kernel space, attach and map the buffer:
  ```c
  struct dma_buf *dmabuf = dma_buf_get(fd);
  struct dma_buf_attachment *attach = dma_buf_attach(dmabuf, dev);
  struct sg_table *sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
  // Program hardware DMA descriptors with sgt->sgl addresses
  ```

---

## 5. Synchronization, Cache Coherency & Metadata

### A. Explicit Synchronization (`sync_file` / `dma_fence`)
* Avoid synchronous waiting or polling.
* Pass release/acquire fence file descriptors (`sync_file`) alongside the `dma_buf_fd`.
* When a hardware block finishes execution, it signals its fence FD, allowing downstream IP blocks to start without CPU intervention.

### B. CPU Access & Cache Coherency
If the CPU needs to read or write to a mapped `dma-buf` in user space (`mmap`), wrap the access with cache synchronization ioctls:

```c
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

struct dma_buf_sync sync_args;

// Before CPU read/write
sync_args.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
ioctl(dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync_args);

// Perform CPU access...

// After CPU read/write
sync_args.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
ioctl(dma_buf_fd, DMA_BUF_IOCTL_SYNC, &sync_args);
```

### C. Metadata Management
`dma-buf` file descriptors do not encode graphics metadata. The vendor pipeline must transport a companion metadata structure (or use `native_handle_t`):
* Width, Height, Stride (per plane)
* Pixel Format (FourCC / Android PixelFormat)
* Color Space / Primaries
* Plane offsets and byte sizes

---

## 6. Bridging Vendor `dma-buf` to Android Framework

When vendor buffers need to interact with Android Framework (e.g., rendering on `SurfaceFlinger` or passing to `MediaCodec`):

```
+---------------------+      Import into Gralloc      +---------------------------+
| Vendor dma-buf FD   |  -------------------------->  | buffer_handle_t           |
| + Layout Metadata   |     via IMapper / Gralloc     | (Usable by Framework/NDK) |
+---------------------+                               +---------------------------+
```

1. **Pack into `native_handle_t`:** Wrap the `dma_buf_fd` in the `fds[]` array and layout metadata in `data[]`.
2. **Import via Gralloc Mapper (`IMapper`):** Use Gralloc's `importBuffer()` to validate and register the handle into the framework process.
3. **Wrap as `AHardwareBuffer`:** In native framework code, `AHardwareBuffer_fromHardwareBuffer()` or internal GraphicBuffer constructors can encapsulate the imported `buffer_handle_t`.

---

## 7. Unified Dual-Mode Interface Design (NDK & Non-NDK / Vendor)

To support both NDK (applications, framework-tied services) and Non-NDK (pure vendor HALs, daemons, Linux kernel accelerators) transparently, a unified buffer abstraction layer is required.

### A. Architectural Interface Blueprint

```
                     +---------------------------------------+
                     |         Unified Buffer Handle         |
                     |  - Metadata (width, height, stride)   |
                     |  - Sync Fence FDs (acquire / release) |
                     +---------------------------------------+
                                    |         |
           +------------------------+         +-----------------------+
           | (Backend: NDK)                                           | (Backend: Non-NDK)
           v                                                          v
+-------------------------------+                         +-------------------------------+
| NDK AHardwareBuffer Adapter   |                         | Raw DMA-BUF Adapter           |
| - Wraps `AHardwareBuffer*`    |                         | - Holds `int dma_buf_fd`      |
| - Lock via `AHardwareBuffer_` |                         | - Allocates: `libdmabufheap`  |
| - Bridged to Gralloc / Mapper |                         | - Sync: `DMA_BUF_IOCTL_SYNC`  |
+-------------------------------+                         +-------------------------------+
```

### B. Core Interface Definitions (C/C++ Header)

```cpp
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__ANDROID__) && defined(SCALIX_USE_NDK_AHARDWAREBUFFER)
#include <android/hardware_buffer.h>
#endif

namespace scalix {

enum class BufferBackendType : uint32_t {
    RAW_DMABUF = 0,         // Non-NDK mode (Raw Linux DMA-BUF FD + libdmabufheap)
    AHARDWARE_BUFFER = 1    // NDK mode (AHardwareBuffer* wrapper)
};

enum class BufferFormat : uint32_t {
    RGBA_8888,
    RGBX_8888,
    RGB_888,
    RGB_565,
    YUV_420_888,
    NV12,
    NV21,
    RAW_OPAQUE,
    BLOB
};

struct BufferDescriptor {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t stride{0};
    BufferFormat format{BufferFormat::RGBA_8888};
    uint64_t usage_flags{0}; // Shared usage bitflags (CPU_READ, CPU_WRITE, GPU, CAMERA)
    size_t size_bytes{0};
    const char* heap_name{"system"}; // Used for DMA-BUF Heaps allocation
};

struct BufferPlaneInfo {
    void* vaddr{nullptr};
    size_t offset{0};
    size_t row_stride{0};
    size_t plane_size{0};
};

struct BufferMapping {
    void* addr{nullptr};
    size_t size{0};
    uint32_t num_planes{1};
    BufferPlaneInfo planes[4];
};

class IUnifiedBuffer {
public:
    virtual ~IUnifiedBuffer() = default;

    // Backend and identity inspection
    virtual BufferBackendType GetBackendType() const = 0;
    virtual const BufferDescriptor& GetDescriptor() const = 0;

    // Primary handle exports
    virtual int GetDmaBufFd() const = 0; // Returns valid FD in both modes (borrowed or extracted)
#if defined(__ANDROID__) && defined(SCALIX_USE_NDK_AHARDWAREBUFFER)
    virtual AHardwareBuffer* GetNativeHardwareBuffer() const = 0; // Nullptr if RAW_DMABUF backend
#endif

    // Synchronization & CPU Mapping
    virtual bool Lock(uint64_t usage, BufferMapping* out_mapping, int acquire_fence_fd = -1) = 0;
    virtual bool Unlock(int* out_release_fence_fd = nullptr) = 0;

    // Sync file fence management
    virtual void SetAcquireFence(int fence_fd) = 0;
    virtual int ExtractReleaseFence() = 0;
};

// Factory functions
class UnifiedBufferFactory {
public:
    // 1. Allocate buffer using the selected backend
    static IUnifiedBuffer* Allocate(const BufferDescriptor& desc, BufferBackendType backend);

    // 2. Wrap existing raw DMA-BUF FD (Non-NDK vendor path)
    static IUnifiedBuffer* CreateFromDmaBuf(int dma_buf_fd, const BufferDescriptor& desc, bool own_fd);

#if defined(__ANDROID__) && defined(SCALIX_USE_NDK_AHARDWAREBUFFER)
    // 3. Wrap existing AHardwareBuffer (NDK path)
    static IUnifiedBuffer* CreateFromAHardwareBuffer(AHardwareBuffer* ahb, bool acquire_ref);
#endif
};

} // namespace scalix
```

### C. Implementation Strategy per Backend

1. **Non-NDK Backend (`RAW_DMABUF`):**
   * **Allocation:** `BufferAllocator::Alloc(desc.heap_name, desc.size_bytes)` (or `open("/dev/dma_heap/...")`).
   * **CPU Mapping (`Lock`):** Performs `mmap()` on `dma_buf_fd` and invokes `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_START`.
   * **CPU Unmap (`Unlock`):** Invokes `DMA_BUF_IOCTL_SYNC` with `DMA_BUF_SYNC_END`.
   * **Zero NDK Dependency:** Compiled into standalone vendor libraries using only Linux headers (`<linux/dma-buf.h>`, `<sys/ioctl.h>`).

2. **NDK Backend (`AHARDWARE_BUFFER`):**
   * **Allocation:** `AHardwareBuffer_allocate(&ahb_desc, &ahb_ptr)`.
   * **CPU Mapping (`Lock`):** `AHardwareBuffer_lock(ahb, usage, acquire_fence, nullptr, &vaddr)`.
   * **CPU Unmap (`Unlock`):** `AHardwareBuffer_unlock(ahb, &release_fence)`.
   * **FD Interop:** On platforms where extraction is supported, bridged to underlying native handles (`buffer_handle_t` -> `int fd`).

---

## 8. Android DMA-BUF vs. Standard Mainline Linux DMA-BUF

While Android utilizes upstream Linux kernel `dma-buf` under the hood, key differences exist across allocation, metadata, synchronization, and ecosystem conventions.

| Architectural Dimension | Android DMA-BUF Ecosystem | Standard Mainline Linux DMA-BUF |
| :--- | :--- | :--- |
| **Allocator Subsystem** | **DMA-BUF Heaps (`/dev/dma_heap/*`)** via `libdmabufheap` (Android 11+ GKI). Legacy devices used `/dev/ion`. | **DRM Dumb Buffers** (`DRM_IOCTL_MODE_CREATE_DUMB`), V4L2 (`VIDIOC_REQBUFS` / `EXPBUF`), `udmabuf`, or mainline `/dev/dma_heap/*`. |
| **Metadata & Layout** | **Gralloc / `native_handle_t` & `AHardwareBuffer`:** Layout metadata (width, height, stride, HAL format, usage) is managed via Gralloc/AIDL Mapper. | **DRM Format Modifiers:** Formats and tiling layouts are described via DRM FourCC codes + 64-bit format modifiers (`DRM_FORMAT_MOD_*` / `EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT`). |
| **Synchronization Model** | **Strict Explicit Sync:** Fences are passed as distinct `sync_file` file descriptors across Binder/AIDL IPC boundaries without blocking. | **Historically Implicit Sync, Moving to Explicit Sync:** Traditionally used implicit DMA fences attached to `dma_resv`; modern Wayland/KMS uses explicit sync via `sync_file` or DRM syncobj (`DRM_IOCTL_SYNCOBJ_*`). |
| **User Space Abstraction** | `AHardwareBuffer` (NDK) / `GraphicBuffer` (Framework) / `libdmabufheap` (Vendor). | `libgbm` (Generic Buffer Management), Wayland `linux-dmabuf-unstable-v1`, V4L2 user space. |
| **Security & Partitioning** | Enforced **SELinux** domain isolation per partition (`/dev/dma_heap/*` access restricted to specific vendor/system domains); AIDL `ParcelFileDescriptor`. | Standard POSIX file permissions, group access (`video`, `render`), standard Unix domain socket `SCM_RIGHTS` FD passing. |
| **Cache Management** | `DMA_BUF_IOCTL_SYNC` ioctls directly, or encapsulated inside `AHardwareBuffer_lock/unlock` and Gralloc mapper locks. | `DMA_BUF_IOCTL_SYNC` (`<linux/dma-buf.h>`) ioctl called around CPU `mmap` accesses. |

---

## 9. Scalix Engine Architecture & Implementation Plan

### A. Dual-Mode Architecture in Scalix Core

The Scalix DMA subsystem is structured to dynamically support both **NDK applications** and **Vendor HALs/drivers** across 64-bit and 32-bit ARM architectures:

```
                                 +-----------------------------------+
                                 |         scalix::DmaBuffer         |
                                 +-----------------------------------+
                                                   |
              +------------------------------------+------------------------------------+
              |                                                                         |
              v                                                                         v
+--------------------------------------------+            +--------------------------------------------+
| NDK Mode (`DmaAllocatorType::AndroidAhb`)  |            | Vendor Mode (`DmaAllocatorType::DmaHeap`)  |
| - Backed by `AHardwareBuffer*`             |            | - Backed by `/dev/dma_heap/*` or raw FD    |
| - Locked via `AHardwareBuffer_lock`        |            | - Mapped via `mmap()` & `DMA_BUF_IOCTL_SYNC`|
| - Zero-copy interop with Camera/Vulkan     |            | - Standalone POSIX/Linux UAPI (no NDK link)|
| - Exports `ahb_handle` and underlying FD   |            | - Imported via `from_raw_dma_buf(fd, ...)` |
+--------------------------------------------+            +--------------------------------------------+
```

### B. Scalix C-API & C++ Binding Additions
1. **Raw DMA-BUF Import:**
   * `scalix_dma_buffer_from_fd(int fd, uint32_t width, uint32_t height, size_t stride_bytes, ScalixPixelFormat format)`
   * Enables hardware vendor blocks (V4L2, DRM, custom DSP) to pass preexisting `dma_buf_fd` directly to Scalix without memory duplication.
2. **Native Handle Inspection:**
   * `scalix_dma_buffer_get_ahb_handle(const ScalixDmaBuffer* buffer)`
   * Returns `AHardwareBuffer*` in NDK mode or `NULL` in pure vendor mode.
3. **Cache Synchronization:**
   * `scalix_dma_buffer_sync_start()` and `scalix_dma_buffer_sync_end()` invoke `DMA_BUF_IOCTL_SYNC` seamlessly across Linux and Android.

### C. Build & Packaging Plan on GitHub Actions CI (`build.yml`)
* **Toolchain:** Android NDK `r27c` pinned with target platform `34` (Android 14+).
* **Matrix Product:**
  * `aarch64-linux-android` $\times$ `[ndk, vendor]`
  * `armv7-linux-androideabi` $\times$ `[ndk, vendor]`
* **Artifact Deliverables:**
  * `scalix-<ver>-android-aarch64.tar.gz` & `.tar.zst`
  * `scalix-<ver>-android-aarch64-vendor.tar.gz` & `.tar.zst`
  * `scalix-<ver>-android-armv7.tar.gz` & `.tar.zst`
  * `scalix-<ver>-android-armv7-vendor.tar.gz` & `.tar.zst`

---

## 10. Action Checklist for Implementation

- [x] **Kernel / Platform Baseline:** Standardize on Linux DMA-BUF Heaps (`/dev/dma_heap/*`) and Android 14+ (API 34). Deprecate legacy ION and HIDL Gralloc.
- [x] **Scalix Core DMA:** Enable `linux_dma_heap` allocator on both Linux and Android (`target_os = "android"`).
- [x] **Raw FD Importer:** Implement `DmaBuffer::from_raw_dma_buf` for vendor zero-copy pipeline integration.
- [x] **C-API Bindings:** Expose `scalix_dma_buffer_from_fd` and `scalix_dma_buffer_get_ahb_handle` in C header (`scalix.h`) and C++ wrapper (`scalix.hpp`).
- [x] **CI Matrix:** Configure Cartesian product matrix in `.github/workflows/build.yml` supporting 32-bit & 64-bit ARM across NDK and Vendor targets.
