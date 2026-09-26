# Scalix

[English](README.md) | [한국어](README.ko.md)

**Scalix** is an extensible, high-performance, hardware-accelerated image scaling and resampling engine engineered for modern **Linux (x86_64)** and **Android (aarch64, API 33+/14+)** platforms (compiled library).

Designed with a **headless-first and offscreen-first** architecture, Scalix provides a unified interface across **Vulkan Compute**, **OpenGL / GLES (EGL Headless)**, **OpenCL (Direct Compute)**, **NPU / Neural Accelerators**, and **Dedicated 2D HW Engines (V4L2 M2M / DRM)**. CPU fallback and host SIMD operations are delegated to third-party image processing libraries (e.g. OpenCV).

---

## Key Features

* **Headless & Offscreen Native:** No display server (X11 / Wayland) required; executes pure offscreen GPU workloads across Vulkan, OpenGL/EGL, and OpenCL in background workers, cloud servers, and embedded pipelines.
* **Unified Multi-Backend Hardware Acceleration:**
  * **Vulkan Backend (`Backend::Vulkan`):** Modern explicit GPU control with fixed-function 2D `Blit`, programmable `Raster` graphics, hierarchical anti-aliasing `LodPyramid` (multi-pass box filtering downscaler), and direct `Compute` shaders.
  * **OpenGL / GLES Backend (`Backend::OpenGL`):** Universal offscreen acceleration via EGL Headless and Framebuffer Objects (FBO), supporting GLES 3.1+ Compute and Framebuffer blits for environments without Vulkan.
  * **OpenCL Backend (`Backend::OpenCL`):** Pure direct compute kernel pipeline (`clEnqueueNDRangeKernel`) with dynamic runtime loading (`dlopen`/`dlsym`), ideal for heterogeneous compute environments, headless servers, and Android OpenCL runtimes.
  * **Dedicated 2D HW & NPU (Extensible):** Scalable engine architecture prepared for zero-copy DMA-BUF hardware scaling (V4L2 M2M, DRM) and neural network pre-processing accelerators.
* **GPU Compute Shader Acceleration:** Native compute pipelines (`VulkanRgbCompute`, `OpenClComputeResizer`, OpenGL CS) for direct GPU-side packed RGB888 unpack/repack and high-order resamplers without CPU memory conversion bottlenecks.
* **Ring-Buffered Staging Allocator:** Triple-buffered staging rings (`VulkanStagingRing`, `GlStagingRing`, `OpenClStagingRing`) with per-slot fence/event synchronization and automatic hysteresis memory management (<50% capacity threshold).
* **Flexible Execution Models:**
  * **Synchronous (Blocking):** Direct execution for CLI tools and deterministic pipelines.
  * **Asynchronous (Zero-Copy Task):** Non-blocking polling and timeout waits with zero-copy descriptor dispatch (`resize_async_raw`).
  * **Callback-Driven:** Event-driven frame completion callbacks dispatched to a background thread pool for streaming, camera, and UI pipelines.
* **Strict 64-Byte Memory Alignment:** 64-byte buffer alignment (`SCALIX_REQUIRED_ALIGNMENT_BYTES = 64`) across all descriptors, enabling optimal AVX-512 / ARM Neon SIMD vectorization and DMA-BUF hardware compatibility.
* **Zero-Copy Memory Subsystem:** First-class support for Linux **DMA-BUF** and Android **AHardwareBuffer** & **DMA-BUF Heaps (`/dev/dma_heap/*`)** across GPU, 2D hardware blitters, and V4L2.
* **Dual-Mode Android Architecture:** Seamless support for both **NDK Mode** (`AHardwareBuffer` for apps) and **Vendor Mode** (raw `/dev/dma_heap/*` and file descriptor importing for HALs/daemons).
* **Multi-Language APIs:** Core engine with stable **C ABI** (`libscalix.so` / `scalix.h`), idiomatic **C++20** wrapper (`scalix.hpp`), and native **Rust** crate.
* **Comprehensive Filter Suite:** Nearest Neighbor, Bilinear, Bicubic (Catmull-Rom 4×4), Lanczos-3 (3-lobe sinc 6×6), Area (pixel box relation), and hierarchical LoD mipchain downscaling.

---

## Compatibility & Verification Matrix

This matrix tracks the hardware backends, memory subsystems, and platform capabilities supported by Scalix, along with their active verification status.

#### Legend
* `✔` **Verified & Tested:** Fully implemented and validated with automated test suite and benchmarks on the test environment.
* `◐` **Compiled / In Progress:** Cross-compiled or library build validated; runtime execution pending physical hardware test.
* `○` **Planned / Unverified:** Supported by architectural specification, pending implementation and test verification.
* `—` **Deferred / N/A:** Planned for future milestone or not applicable for the target platform.

---

### 1. Hardware Backends & Accelerators

| Backend Provider | Subsystem / API | Host / Platform Target | WSL2 (Ubuntu 24.04, NVIDIA GPU) | Linux (x86_64 Bare-Metal) | Android (aarch64, API 34+) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Vulkan Offscreen** | Graphics (`Blit`, `Raster`, `LodPyramid`, `Compute`) | Modern GPU (Vulkan 1.1+) | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **OpenGL / GLES** | EGL Headless / FBO / CS (`Blit`, `Raster`, `LodPyramid`, `Compute`) | GLES 3.1+ / GL 4.3+ / Mesa | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **OpenCL** | Direct Compute (`clEnqueueNDRangeKernel`) | OpenCL 1.2+ / 3.0 (Linux / Android) | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Linux 2D HW Engines | ○ Mock / Loopback | ○ Hardware Req. | — |
| **NPU / AI Engine** | QNN / OpenVINO / Vendor NPU | Neural Accelerators | ○ Mock / CPU | ○ OpenVINO | ○ Supported (Unverified) |

---

### 2. Memory & Zero-Copy Subsystems

| Feature | Interface / Handle | WSL2 (Ubuntu 24.04, NVIDIA GPU) | Bare-Metal Linux (x86_64) | Android (API 33+/14+, aarch64) |
| :--- | :--- | :---: | :---: | :---: |
| **64-Byte Aligned Host Memory** | Contiguous 64-byte aligned CPU memory (RGB/RGBA) | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **Triple-Buffered Staging Ring** | 3-slot pinned / mapped staging buffer ring with hysteresis | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **Linux & Android DMA-BUF** | `dma_buf_fd` (DMA-BUF Heaps `/dev/dma_heap/*` zero-copy) | ◐ Fallback (Verified) | ○ Supported (Unverified) | ◐ Compiled Library (Unverified) |
| **Raw DMA-BUF Import** | `from_fd(fd, ...)` / `scalix_dma_buffer_from_fd` | ✔ Verified | ◐ Compiled (Unverified) | ◐ Compiled Library (Unverified) |
| **Android AHardwareBuffer** | `AHardwareBuffer*` zero-copy interop (NDK mode) | — | — | ◐ Compiled Library (Unverified) |

> [!NOTE]
> **Current Verification & Target Platform Status**
> - **WSL2 (Ubuntu 24.04, NVIDIA GPU):** The primary and currently tested development platform. Offscreen Vulkan rendering is verified via `/dev/dxg` on NVIDIA GPU. Linux `dma-buf` automatically falls back to 64-byte aligned host staging memory.
> - **Linux (x86_64 Bare-Metal):** Compiled and architected for native Linux execution (DMA-Heap and DRM GEM dumb buffer support), but not yet verified on physical bare-metal hardware.
> - **Android (aarch64, API 34+):** Android NDK cross-compilation (`cargo-ndk`) and compiled library generation (`libscalix.so` / `libscalix.a`) are supported across **NDK** and **Vendor** variants. Standalone vendor mode operates on pure Linux DMA-BUF Heaps without NDK runtime dependencies. Physical hardware testing is pending.

---

### 3. Pixel Format & Filter Strategy Matrix (Vulkan Backend)

| Pixel Format (In / Out) | `Nearest` | `Bilinear` | `Bicubic` | `Lanczos3` | `Area` | `LodPyramid` (Downscale) | Default `Auto` Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RGB888` / `BGR888` (24-bit)** | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Raster` / `Compute` (✔) | **`Compute`** (Fused 24-bit pass) |
| **`RGBA8888` / `BGRA8888` (32-bit)** | `Blit` / `Compute` (✔) | `Blit` / `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `LodPyramid` (✔) | **`Blit`** (Fast-path) / **`Compute`** (High-Order) |
| **`R8` / `RG88` (Single/Dual Ch)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`RGBA16F` / `RGBA32F` (HDR/Float)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`NV12` / `YUV420p` (Semi/Planar)** | `Raster` / `Blit` (◐) | `Raster` / `Blit` (◐) | — | — | — | — | **`Raster`** (Y/UV planar pass) |

> [!NOTE]
> **Filter Algorithm Reference Implementations**
> - **Bicubic (`FilterMode::Bicubic`):** Implements 2D separable Catmull-Rom cubic spline interpolation (`a = -0.5`) across a 4×4 tap neighborhood ([Keys, 1981](https://doi.org/10.1109/TASSP.1981.1163711)).
> - **Lanczos3 (`FilterMode::Lanczos3`):** Implements 2D separable 3-lobe sinc-windowed sinc filtering (`a = 3`, `L(x) = sinc(x) · sinc(x/3)`) across a 6×6 tap window with dynamic weight normalization to prevent DC energy drift ([Lanczos, 1956](https://archive.org/details/appliedanalysis0000corn); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **Area (`FilterMode::Area`):** Implements pixel area relation / box averaging with exact subpixel 2D bounding area overlap integration across source texels, preserving total pixel energy without aliasing ([Crow, 1984](https://doi.org/10.1145/964965.808599); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **Strategy Routing (`VulkanStrategy::Auto`):** Direct compute kernels are selected for packed 24-bit (`RGB888` / `BGR888`) to avoid CPU-host expansion bottlenecks. For 32-bit `RGBA8888`, hardware fixed-function `Blit` is preferred for `Nearest` and `Bilinear` workloads for maximum raw fill-rate throughput, while `Bicubic`, `Lanczos3`, and `Area` dispatch to GPU `Compute`.

---

### 4. Cross-Backend Filter & Execution Strategy Categorization

Scalix provides unified, standardized filter mode abstractions while leveraging the diverse hardware execution pipelines available across GPU and compute runtimes:

| Category | Filter Mode | Mathematical / Algorithmic Model | Vulkan Execution Paths | OpenGL / EGL Execution Paths | OpenCL Execution Paths |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Nearest** | `FilterMode::Nearest` | 0-Order Hold (Nearest Neighbor) | `Blit`, `Raster`, `Compute` | `Blit`, `Raster`, `Compute` | `Compute` |
| **Bilinear** | `FilterMode::Bilinear` | 1st-Order Tent Filter (Linear) | `Blit`, `Raster`, `Compute` | `Blit`, `Raster`, `Compute` | `Compute` |
| **Hierarchical LoD** | `Strategy::LodPyramid` | Multi-Pass Mipchain Reduction | `Lod Full`, `Lod 2-Pass` | `Lod Full`, `Lod 2-Pass` | — *(Graphics HW Only)* |
| **Bicubic** | `FilterMode::Bicubic` | Catmull-Rom 4×4 Spline ($a = -0.5$) | `Compute` | `Compute` | `Compute` |
| **Lanczos-3** | `FilterMode::Lanczos3` | 3-Lobe Sinc Windowed ($6\times 6$, $a = 3$) | `Compute` | `Compute` | `Compute` |
| **Area** | `FilterMode::Area` | 2D Continuous Subpixel Box Overlap | `Compute` | `Raster`, `Compute` | `Compute` |
| **Adaptive Auto** | `Strategy::Auto` | Engine Fast-Path Dynamic Selector | `Auto` | `Auto` | `Auto` |

> [!TIP]
> **Graphics vs. Direct Compute Implementation Highlights**
> - **Hierarchical LoD Downscaling:** Available exclusively on graphics backends (Vulkan / OpenGL), exploiting hardware texture mipchain generation (`vkCmdBlitImage` / `glGenerateMipmap`) for aliasing-free multi-octave reduction.
> - **Area Box Averaging:** Executed via direct 2D integration compute kernels across all backends (`VulkanComputeResizer`, `GlComputeResizer`, `OpenClComputeResizer`), with OpenGL also offering dedicated fragment shader rasterization (`raster_area.frag`).

---

## Quick Start & API Preview

### 1. Basic Image Resize (C++20 & Pure C ABI)

#### C++20 API (`<scalix/scalix.hpp>`)

```cpp
#include <scalix/scalix.hpp>

int main() {
    // 1. Initialize Scalix Engine (Auto selects Vulkan hardware backend)
    scalix::Engine engine(scalix::Backend::Auto);

    scalix::ImageDesc src{
        .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = src_ptr,
        .data_len = 3840 * 2160 * 4, .dma_buf_fd = -1
    };

    scalix::ImageDesc dst{
        .width = 320, .height = 320, .stride_bytes = 320 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = dst_ptr,
        .data_len = 320 * 320 * 4, .dma_buf_fd = -1
    };

    // Configure dynamic resize options (Strategy: Blit, Raster, LodPyramid, Compute)
    const scalix::ResizeOptions options = scalix::ResizeOptions::with_vulkan(
        scalix::Filter::Bilinear,
        scalix::Strategy::LodPyramid,
        2 // Hierarchical anti-aliased downscale (max mip levels = 2)
    );

    // Mode A: Synchronous (Blocking)
    engine.resize(src, dst, options);

    // Mode B: Asynchronous Task (Zero-Copy handle polling & timeout wait)
    auto task = engine.resize_async(src, dst, options);
    task.wait(1000); // Wait up to 1000ms for completion

    // Mode C: Callback-driven (Dispatched to worker thread pool)
    engine.resize_callback(src, dst, options, [](int status) {
        // Handle completion asynchronously
    });

    return 0;
}
```

#### Pure C ABI (`<scalix/scalix.h>`)

```c
#include <scalix/scalix.h>

ScalixEngine* engine = scalix_engine_create(SCALIX_BACKEND_AUTO);

ScalixImageDesc src = { .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
                        .format = SCALIX_FORMAT_RGBA8888, .host_ptr = src_ptr,
                        .data_len = 3840 * 2160 * 4, .dma_buf_fd = -1 };
ScalixImageDesc dst = { .width = 320, .height = 320, .stride_bytes = 320 * 4,
                        .format = SCALIX_FORMAT_RGBA8888, .host_ptr = dst_ptr,
                        .data_len = 320 * 320 * 4, .dma_buf_fd = -1 };

ScalixVulkanOptions vk_opts = {
    .header = {
        .backend_type = SCALIX_BACKEND_VULKAN,
        .struct_size = sizeof(ScalixVulkanOptions),
    },
    .strategy = SCALIX_STRATEGY_LOD_PYRAMID,
    .max_mip_levels = 2,
};

ScalixResizeOptions options = {
    .filter = SCALIX_FILTER_BILINEAR,
    .backend_options = &vk_opts.header,
};

// Synchronous resize with dynamic options
scalix_resize_sync_with_options(engine, &src, &dst, &options);

scalix_engine_destroy(engine);
```

---

### 2. Advanced: Zero-Copy Hardware DMA Buffers (`DmaBuffer`)

Allocate hardware DMA memory (Linux DMA-Heap / DRM GEM Dumb or Android `AHardwareBuffer`) and execute scaling without intermediate CPU-GPU staging copies:

```cpp
#include <scalix/scalix.hpp>
#include <iostream>

void process_dma_pipeline(scalix::Engine& engine) {
    constexpr uint32_t src_w = 3840, src_h = 2160;
    constexpr uint32_t dst_w = 320, dst_h = 320;

    // 1. Allocate hardware DMA buffers (Auto: DMA-Heap -> DRM GEM Dumb -> Android AHB)
    scalix::DmaBuffer src_dma(src_w, src_h, scalix::PixelFormat::Rgba8888);
    scalix::DmaBuffer dst_dma(dst_w, dst_h, scalix::PixelFormat::Rgba8888);

    // 2. Safe CPU write: with_write handles sync_start(true) and sync_end(true) RAII
    src_dma.with_write([](uint8_t* host_ptr, size_t size) {
        // Decode camera/video frame directly into mapped DMA memory
    });

    // 3. Obtain zero-copy image descriptors referencing DMA file descriptors (fd)
    auto src_desc = src_dma.as_image_desc();
    auto dst_desc = dst_dma.as_image_desc();

    // 4. Execute GPU scaling directly on hardware DMA buffers
    const auto options = scalix::ResizeOptions::with_vulkan(
        scalix::Filter::Lanczos3,
        scalix::Strategy::Compute
    );
    engine.resize(src_desc, dst_desc, options);

    // 5. Safe CPU read: with_read handles sync_start(false) and sync_end(false) RAII
    dst_dma.with_read([](const uint8_t* host_ptr, size_t size) {
        // Access scaled frame with invalid cache lines refreshed
    });
}
```

---

### 3. Advanced: GPU Latency Profiling & Custom Thread Naming

Enable zero-overhead GPU hardware timestamp profiling and configure dedicated Linux thread names (`<prefix>/scx-hw`, `<prefix>/scx-w<id>`):

```cpp
#include <scalix/scalix.hpp>
#include <iostream>

void run_profiled_resizer() {
    // Initialize engine with custom thread prefix (enforces Linux 15-char comm limit)
    scalix::Engine engine(scalix::Backend::Auto, "infer");

    // Enable hardware timestamp query pools and stage latency tracking
    engine.set_profiling(true);

    // Execute resize...
    // engine.resize(src, dst, options);

    // Query high-precision execution metrics
    if (const auto p = engine.last_profile()) {
        std::cout << "[Scalix Hardware Latency Breakdown]\n";
        std::cout << "  Host Unpack       : " << p->host_unpack_ms << " ms\n";
        std::cout << "  GPU Staging Upload: " << p->gpu_upload_ms << " ms\n";
        std::cout << "  GPU Core Scaling  : " << p->gpu_pure_blit_ms << " ms\n";
        std::cout << "  GPU Readback      : " << p->gpu_download_ms << " ms\n";
        std::cout << "  Driver/HW Sync    : " << p->driver_sync_ms << " ms\n";
        std::cout << "  Total Wall-Clock  : " << p->total_wall_ms << " ms\n";
    }
}
```

---

## Building & Testing

### Prerequisites
* **Rust Toolchain:** `rustc` & `cargo` (1.70+ recommended)
* **C/C++ Toolchain:** `g++` or `clang++` supporting C++20
* **Image Codec Libraries:** `libjpeg-dev` / `libjpeg-turbo8-dev` (for JPEG I/O examples)
* **Benchmark & Comparison Libraries:** `libopencv-dev` (required for benchmarks comparing Scalix execution latency against OpenCV CPU/GPU operations)
* **GPU Backend Libraries:**
  * **Vulkan:** `libvulkan-dev`, `vulkan-tools`, `mesa-vulkan-drivers`
  * **OpenGL / GLES:** `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libgl1-mesa-dev`
  * **OpenCL:** `ocl-icd-libopencl1`, `mesa-opencl-icd`, `pocl-opencl-icd`
* **Android Cross-Compilation (Optional):** Android NDK (r27+ recommended, API Level 26+, `aarch64-linux-android`) and `cargo-ndk`

> [!NOTE]
> `libopencv-dev` is required for benchmarking workloads (`cpp_benchmark_vulkan`, `cpp_benchmark_gl`, `cpp_benchmark_opencl`) to provide side-by-side execution time comparisons between Scalix hardware pipelines and standard OpenCV image processing operations (`cv::resize`, `cv::warpAffine`).

### 1. Build Rust Core & C ABI Library
To build the static/shared library (`libscalix.so` / `libscalix.a`):
```bash
# Debug build
cargo build

# Optimized release build (generates target/release/libscalix.so)
cargo build --release
```

### 2. Run Test Suite
To execute all unit, integration, and multi-threaded callback tests:
```bash
cargo test
```

### 3. Build & Run C++ Examples
Examples are managed via [`examples/Makefile`](examples/Makefile). You can build and run them directly from the workspace root:

```bash
# Build all examples (automatically builds Cargo workspace if needed)
make -C examples

# Run all examples (including sample.jpg JPEG processing with libjpeg-turbo)
make -C examples run

# Run all performance benchmarks (all available on platform: Vulkan, OpenGL/EGL, OpenCL)
make -C examples benchmark

# Run multi-resolution performance benchmark (Vulkan)
make -C examples benchmark_vulkan

# Run multi-resolution performance benchmark (OpenGL/EGL)
make -C examples benchmark_gl

# Run multi-resolution performance benchmark (OpenCL)
make -C examples benchmark_opencl

# Clean example build artifacts
make -C examples clean
```

#### Individual Examples:
* **`cpp_basic`**: Demonstrates synchronous, asynchronous callback, and zero-copy DMA buffer pre-allocation.
* **`cpp_jpeg`**: Loads [`assets/sample.jpg`](assets/sample.jpg) using `libjpeg-turbo`, decodes directly into memory-mapped DMA buffers, executes the Scalix pipeline (`blit`, `raster`, or `lod [max_mip_levels]`), and writes the output JPEG.
* **`cpp_benchmark_vulkan`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors using the **Vulkan** backend.
* **`cpp_benchmark_gl`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors using the **OpenGL / EGL** backend.
* **`cpp_benchmark_opencl`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors using the **OpenCL** backend.

---

## Zero-Copy DMA Subsystem by Platform

Scalix provides unified zero-copy DMA buffer allocation across supported target environments:

### 1. Bare-Metal Linux (x86_64, Kernel 5.6+ - Unverified)
* **Allocators:** Uses **DMA-Heap** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`) with fallback to **DRM Render Node Dumb Buffers** (`/dev/dri/renderD128` via GEM PRIME).
* **Raw FD Import:** Supports wrapping external DMA-BUF file descriptors via `DmaBuffer::from_fd()`.
* **Permissions:** Ensure the executing user is added to `render` and `video` groups:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### 2. WSL2 (Windows Subsystem for Linux 2: Ubuntu 24.04, NVIDIA GPU)
* **GPU Acceleration:** Fully supported via Microsoft DirectX bridge (`/dev/dxg`) and Mesa Vulkan/D3D12 for offscreen rendering (currently tested platform).
* **DMA Allocation Behavior & Limitation:** Linux `dma-buf` is **not yet supported** across the `/dev/dxg` virtual translation layer. Stock WSL2 kernels do not provide `/dev/dma_heap`, and DRM GEM dumb buffer allocations often fail or lack PRIME hardware export support. Scalix's runtime allocator probing detects this DMA failure automatically and safely falls back to host memory staging buffers. Testing true zero-copy DMA requires bare-metal Linux with native DRM render nodes.

### 3. Android (API 33+ / 14+, `aarch64`)
* **Dual-Mode Architecture:**
  * **NDK Mode:** Uses **`AHardwareBuffer`** (`AHardwareBuffer_allocate`, `AHardwareBuffer_lock`) for app and framework zero-copy sharing.
  * **Vendor Mode:** Uses **Linux DMA-BUF Heaps** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`) via direct POSIX `ioctl(DMA_HEAP_IOCTL_ALLOC)` with zero external `.so` dependencies (no `libdmabufheap.so` or `dlopen` required).
  * **Zero-Copy FD Import:** External pipelines (Camera V4L2, DRM, custom DSP/NPU) pass raw `dma_buf_fd` handles directly into `DmaBuffer::from_fd()`.
* **Platform Gating & Build:** Cross-compiled via NDK r27c (Platform 34) for `aarch64-linux-android` in both NDK and Vendor variants.

---

## License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for details.



