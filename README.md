# Scalix

[English](README.md) | [한국어](README.ko.md)

**Scalix** is an extensible, high-performance, hardware-accelerated image scaling and resampling engine engineered exclusively for modern **Linux (x86_64)** and **Android (aarch64 / armv7)** platforms.

Designed with a **headless-first and offscreen-first** architecture, Scalix provides a unified interface across **Vulkan Compute**, **OpenGL / GLES (EGL Headless)**, **NPU / Neural Accelerators**, and **Dedicated 2D HW Engines (V4L2 M2M / DRM)**. CPU fallback and host SIMD operations are delegated to third-party image processing libraries (e.g. OpenCV).

---

## Key Features

* **Headless & Offscreen Native:** No display server (X11 / Wayland) required; executes pure offscreen GPU workloads in background workers, cloud servers, and embedded pipelines.
* **Vulkan Offscreen Rendering Strategies:**
  * **Hardware Blit (`Blit`):** Fixed-function 2D blitting for maximum raw throughput and zero shader overhead.
  * **Offscreen Raster Graphics (`Raster`):** Complete graphics pipeline with vertex/fragment shaders and hardware bilinear/trilinear samplers.
  * **Hierarchical LoD Pyramid (`LodPyramid`):** Multi-pass 2×2 box filtering downscaler with configurable `max_mip_levels` to eliminate aliasing and moiré artifacts during extreme downscaling (>4×).
* **GPU Compute Shader Acceleration:** Native compute pipeline (`VulkanRgbCompute`) for GPU-side packed RGB888 unpack and repack passes, eliminating CPU host memory bottlenecks.
* **Ring-Buffered Staging Allocator:** Triple-buffered staging ring (`VulkanStagingRing`, 3 slots) with per-slot `VkFence` synchronization and automatic memory retention with hysteresis shrinking (<50% threshold) for seamless CPU/GPU pipelining.
* **Flexible Execution Models:**
  * **Synchronous (Blocking):** Direct execution for CLI tools and deterministic pipelines.
  * **Asynchronous (Zero-Copy Task):** Non-blocking polling and timeout waits with zero-copy descriptor dispatch (`resize_async_raw`).
  * **Callback-Driven:** Event-driven frame completion callbacks for streaming, camera, and UI pipelines.
* **Strict 64-Byte Memory Alignment:** 64-byte buffer alignment (`SCALIX_REQUIRED_ALIGNMENT_BYTES = 64`) across all descriptors, enabling optimal AVX-512 / ARM Neon SIMD vectorization and DMA-BUF hardware compatibility.
* **Zero-Copy Memory Subsystem:** First-class support for Linux **DMA-BUF** and Android **AHardwareBuffer** across GPU, 2D hardware blitters, and V4L2.
* **Multi-Language APIs:** Core engine with stable **C ABI** (`libscalix.so` / `scalix.h`), idiomatic **C++20** wrapper (`scalix.hpp`), and native **Rust** crate.
* **Comprehensive Filter Suite:** Nearest Neighbor, Bilinear, Bicubic, and hierarchical mipchain downscaling.

---

## Compatibility & Verification Matrix

This matrix tracks the hardware backends, execution paradigms, and platform capabilities supported by Scalix, along with their active verification status.

#### Legend
* `✔` **Verified & Tested:** Fully implemented and validated with automated test suite and benchmarks.
* `◐` **Compiled / In Progress:** Cross-compiled or validated in CI, pending physical hardware runtime test.
* `○` **Planned / Unverified:** Supported by architectural specification, pending implementation and test verification.
* `—` **Deferred / N/A:** Planned for future milestone or not applicable for the target platform.

---

### 1. Hardware Backends & Accelerators

| Backend Provider | Subsystem / API | Host / Silicon Target | WSL2 Dev Host | Linux (x86_64) | Android (aarch64 / armv7) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Vulkan Offscreen** | Graphics (`Blit`, `Raster`, `LodPyramid`, `Compute`) | Modern GPU (AMD / NVIDIA / Intel / Mesa Lavapipe) | ✔ Verified | ✔ Verified | ◐ Compiled (Unverified) |
| **OpenGL / GLES** | EGL Headless / FBO / CS (`Blit`, `Raster`, `LodPyramid`, `Compute`) | GLES 3.1+ / GL 4.3+ / Mesa | ✔ Verified | ✔ Verified | ◐ Compiled (Unverified) |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Rockchip RGA, NXP PXP, Allwinner G2D | ○ Mock / Loopback | ○ Hardware Req. | — |
| **NPU / AI Engine** | NNAPI / QNN / OpenVINO | Qualcomm HTP, Intel NPU, MediaTek APU | ○ Mock / CPU | ○ OpenVINO | ○ QNN / NNAPI |

---

### 2. Execution Paradigms

| Execution Mode | Description | Rust Core | C ABI | C++20 API |
| :--- | :--- | :---: | :---: | :---: |
| **Synchronous (`sync`)** | Blocking call until GPU completion or timeout | ✔ | ✔ | ✔ |
| **Asynchronous (`async`)** | Zero-copy `TaskHandle` / `std::future` with staging ring overlap | ✔ | ✔ | ✔ |
| **Callback (`callback`)** | Dispatches completion function on worker thread pool | ✔ | ✔ | ✔ |

---

### 3. Memory & Zero-Copy Subsystems

| Feature | Interface / Handle | Bare-Metal Linux (x86_64) | WSL2 (Ubuntu 22.04) | Android (aarch64 / armv7, API 26+) |
| :--- | :--- | :---: | :---: | :---: |
| **64-Byte Aligned Host Memory** | Contiguous 64-byte aligned CPU memory (RGB/RGBA) | ✔ Verified | ✔ Verified | ◐ Compiled (Unverified) |
| **Triple-Buffered Staging Ring** | 3-slot pinned / mapped staging buffer ring with hysteresis | ✔ Verified | ✔ Verified | ◐ Compiled (Unverified) |
| **Linux DMA-BUF** | `dma_buf_fd` (Vulkan / EGL / DRM PRIME zero-copy) | ○ Supported | ◐ Fallback | — |
| **AHardwareBuffer** | `AHardwareBuffer*` zero-copy interop | — | — | ◐ Compiled (Unverified) |

> [!NOTE]
> **Current Verification & Target Platform Status**
> - **Linux (x86_64):** Verified on NVIDIA GPU (via Vulkan driver) and automated CI pipeline with Mesa Lavapipe Vulkan software rasterizer.
> - **WSL2 (Windows Subsystem for Linux 2):** Offscreen Vulkan rendering is verified via `/dev/dxg` on NVIDIA GPU. Linux `dma-buf` automatically falls back to 64-byte aligned host staging memory.
> - **Android (aarch64 / armv7):** Android NDK cross-compilation (`cargo-ndk`) and dynamic library generation are validated in CI. However, runtime GPU execution, Vulkan drivers, and `AHardwareBuffer` zero-copy DMA sharing are **not yet verified** on physical Android hardware or emulators.

---

### 4. Pixel Format & Filter Strategy Matrix (Vulkan Backend)

| Pixel Format (In / Out) | `Nearest` | `Bilinear` | `Bicubic` | `Lanczos3` | `Area` | `LodPyramid` (Downscale) | Default `Auto` Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RGB888` / `BGR888` (24-bit)** | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Raster` / `Compute` (✔) | **`Compute`** (Fused 24-bit pass) |
| **`RGBA8888` / `BGRA8888` (32-bit)** | `Blit` / `Compute` (✔) | `Blit` / `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `LodPyramid` (✔) | **`Blit`** (Fast-path) / **`Compute`** (High-Order) |
| **`R8` / `RG88` (Single/Dual Ch)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`RGBA16F` / `RGBA32F` (HDR/Float)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`NV12` / `YUV420p` (Semi/Planar)** | `Raster` / `Blit` (◐) | `Raster` / `Blit` (◐) | — | — | — | — | **`Raster`** (Y/UV planar pass) |

> [!NOTE]
> **Filter Algorithm Reference Implementations**
> - **Bicubic (`FilterMode::Bicubic`):** Implements 2D separable Catmull-Rom cubic spline interpolation ($a = -0.5$) across a $4 \times 4$ tap neighborhood ([Keys, 1981](https://doi.org/10.1109/TASSP.1981.1163711)).
> - **Lanczos3 (`FilterMode::Lanczos3`):** Implements 2D separable 3-lobe sinc-windowed sinc filtering ($a = 3$, $L(x) = \text{sinc}(x)\text{sinc}(x/3)$) across a $6 \times 6$ tap window with dynamic weight normalization to prevent DC energy drift ([Lanczos, 1956](https://archive.org/details/appliedanalysis0000corn); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **Area (`FilterMode::Area`):** Implements pixel area relation / box averaging with exact subpixel 2D bounding area overlap integration across source texels, preserving total pixel energy without aliasing ([Crow, 1984](https://doi.org/10.1145/964965.808599); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **Strategy Routing (`VulkanStrategy::Auto`):** Direct compute kernels are selected for packed 24-bit (`RGB888` / `BGR888`) to avoid CPU-host expansion bottlenecks. For 32-bit `RGBA8888`, hardware fixed-function `Blit` is preferred for `Nearest` and `Bilinear` workloads for maximum raw fill-rate throughput, while `Bicubic`, `Lanczos3`, and `Area` dispatch to GPU `Compute`.

---

## Quick Start & API Preview

### C++20 API (`<scalix/scalix.hpp>`)

```cpp
#include <scalix/scalix.hpp>

int main() {
    // 1. Initialize Scalix Engine
    scalix::Engine engine(scalix::Backend::Auto);
    engine.set_profiling(true); // Optional hardware latency profiling

    scalix::ImageDesc src{
        .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = src_data
    };

    scalix::ImageDesc dst{
        .width = 320, .height = 320, .stride_bytes = 320 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = dst_data
    };

    // Configure dynamic resize options (Strategy: Blit, Raster, LodPyramid)
    const scalix::ResizeOptions options = scalix::ResizeOptions::with_vulkan(
        scalix::Filter::Bilinear,
        scalix::Strategy::LodPyramid,
        2 // Hierarchical anti-aliased downscale
    );

    // Mode A: Synchronous
    engine.resize(src, dst, options);

    // Mode B: Asynchronous Task
    auto task = engine.resize_async(src, dst, options);
    task.wait(); // Wait for completion

    // Mode C: Callback-driven
    engine.resize_callback(src, dst, options, [](int status) {
        // Handle completion on worker thread pool
    });

    return 0;
}
```

### Pure C ABI (`<scalix/scalix.h>`)

```c
#include <scalix/scalix.h>

ScalixEngine* engine = scalix_engine_create(SCALIX_BACKEND_AUTO);

ScalixImageDesc src = { .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
                        .format = SCALIX_FORMAT_RGBA8888, .host_ptr = src_ptr, .dma_buf_fd = -1 };
ScalixImageDesc dst = { .width = 320, .height = 320, .stride_bytes = 320 * 4,
                        .format = SCALIX_FORMAT_RGBA8888, .host_ptr = dst_ptr, .dma_buf_fd = -1 };

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

## Building & Testing

### Prerequisites
* **Rust Toolchain:** `rustc` & `cargo` (1.70+ recommended)
* **C/C++ Toolchain:** `g++` or `clang++` supporting C++20
* **Image Codec Libraries:** `libjpeg-dev` / `libjpeg-turbo8-dev` (for JPEG I/O examples)
* **Benchmark & Comparison Libraries:** `libopencv-dev` (required for benchmarks comparing Scalix execution latency against OpenCV CPU/GPU operations)
* **GPU Backend Libraries:**
  * **Vulkan (P0):** `libvulkan-dev`, `vulkan-tools`, `mesa-vulkan-drivers`
  * **OpenGL / GLES (P1):** `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libgl1-mesa-dev`
* **Android Cross-Compilation (Optional):** Android NDK (r27+ recommended, API Level 26+) and `cargo-ndk`

> [!NOTE]
> `libopencv-dev` is required for benchmarking workloads (`cpp_benchmark`) to provide side-by-side execution time comparisons between Scalix hardware pipelines and standard OpenCV image processing operations (`cv::resize`, `cv::warpAffine`).

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

# Run all performance benchmarks (all available on platform: Vulkan & OpenGL/EGL)
make -C examples benchmark

# Run multi-resolution performance benchmark (Vulkan)
make -C examples benchmark_vulkan

# Run multi-resolution performance benchmark (OpenGL/EGL)
make -C examples benchmark_gl

# Clean example build artifacts
make -C examples clean
```

#### Individual Examples:
* **`cpp_basic`**: Demonstrates synchronous, asynchronous callback, and zero-copy DMA buffer pre-allocation.
* **`cpp_jpeg`**: Loads [`assets/sample.jpg`](assets/sample.jpg) using `libjpeg-turbo`, decodes directly into memory-mapped DMA buffers, executes the Scalix pipeline (`blit`, `raster`, or `lod [max_mip_levels]`), and writes the output JPEG.
* **`cpp_benchmark`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors using the **Vulkan** backend.
* **`cpp_benchmark_gl`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors using the **OpenGL / EGL** backend.

---

## Zero-Copy DMA Subsystem by Platform

Scalix provides unified zero-copy DMA buffer allocation across supported target environments:

### 1. Bare-Metal Linux (x86_64, Kernel 5.6+)
* **Allocators:** Uses **DMA-Heap** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`) with fallback to **DRM Render Node Dumb Buffers** (`/dev/dri/renderD128` via GEM PRIME).
* **Permissions:** Ensure the executing user is added to `render` and `video` groups:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### 2. WSL2 (Windows Subsystem for Linux 2: Ubuntu 22.04, NVIDIA GPU + AMD CPU)
* **GPU Acceleration:** Fully supported via Microsoft DirectX bridge (`/dev/dxg`) and Mesa Vulkan/D3D12 for offscreen rendering.
* **DMA Allocation Behavior & Limitation:** Linux `dma-buf` is **not yet verified to work** on WSL2. Stock WSL2 kernels do not provide `/dev/dma_heap`, and DRM GEM dumb buffer allocations often fail or lack PRIME hardware export support across the `/dev/dxg` virtual translation layer. Scalix's runtime allocator probing detects this DMA failure automatically and safely falls back to host memory staging buffers. Testing true zero-copy DMA requires bare-metal Linux with native DRM render nodes.

### 3. Android (API Level 26+, `aarch64` / `armv7`)
* **Allocator:** Native **`AHardwareBuffer`** (`AHardwareBuffer_allocate`, `AHardwareBuffer_lock`).
* **Platform Gating:** Compiled for `aarch64-linux-android` and `armv7-linux-androideabi` (`-landroid`) with NDK r27+. Android symbols are never linked or exposed on Linux builds.

---

## License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](file:///home/duty/workspace/scalix/LICENSE) file for details.


