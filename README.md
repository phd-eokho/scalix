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
* **Flexible Execution Models:**
  * **Synchronous (Blocking):** Direct execution for CLI tools and deterministic pipelines.
  * **Asynchronous (Future / Task):** Non-blocking polling and timeout waits with hardware timeline semaphores.
  * **Callback-Driven:** Event-driven frame completion callbacks for streaming, camera, and UI pipelines.
* **In-Process Context Worker:** Embedded worker thread pool managing accelerator context affinity (single-threaded EGL / Vulkan queue ownership) and pipelining memory staging.
* **Zero-Copy Memory Subsystem:** First-class support for Linux **DMA-BUF** and Android **AHardwareBuffer** across GPU, 2D hardware blitters, and V4L2.
* **Multi-Language APIs:** Core engine with stable **C ABI** (`libscalix.so` / `scalix.h`), idiomatic **C++20** wrapper (`scalix.hpp`), and native **Rust** crate.
* **Comprehensive Filter Suite:** Nearest Neighbor, Bilinear, Bicubic, and hierarchical mipchain downscaling.

---

## Compatibility & Verification Matrix

This matrix tracks the hardware backends, execution paradigms, and platform capabilities supported by Scalix, along with their active verification status.

#### Legend
* `✔` **Verified & Tested:** Fully implemented and validated with automated test suite and benchmarks.
* `◐` **In Progress / Scaffolded:** Core interface or backend under active implementation.
* `○` **Planned / Unverified:** Supported by architectural specification, pending implementation and test verification.
* `—` **Deferred / N/A:** Planned for future milestone or not applicable for the target platform.

---

### 1. Hardware Backends & Accelerators

| Backend Provider | Subsystem / API | Host / Silicon Target | WSL2 Dev Host | Linux (x86_64) | Android (aarch64 / armv7) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Vulkan Offscreen** | Graphics (`Blit`, `Raster`, `LodPyramid`) | Modern GPU (AMD / NVIDIA / Intel / Mesa LLVMpipe) | ✔ Verified | ✔ Verified | ○ Supported |
| **OpenGL / GLES** | EGL Headless / FBO / CS | GLES 3.1+ / GL 4.3+ | ○ Supported | ○ Supported | ○ Supported |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Rockchip RGA, NXP PXP, Allwinner G2D | ○ Mock / Loopback | ○ Hardware Req. | — |
| **NPU / AI Engine** | NNAPI / QNN / OpenVINO | Qualcomm HTP, Intel NPU, MediaTek APU | ○ Mock / CPU | ○ OpenVINO | ○ QNN / NNAPI |

---

### 2. Execution Paradigms

| Execution Mode | Description | Rust Core | C ABI | C++20 API |
| :--- | :--- | :---: | :---: | :---: |
| **Synchronous (`sync`)** | Blocking call until GPU completion or timeout | ✔ | ✔ | ✔ |
| **Asynchronous (`async`)** | Returns `TaskHandle` / `std::future` / Rust `Future` | ✔ | ✔ | ✔ |
| **Callback (`callback`)** | Dispatches completion function on worker thread pool | ✔ | ✔ | ✔ |

---

### 3. Memory & Zero-Copy Subsystems

| Feature | Interface / Handle | Bare-Metal Linux (x86_64) | WSL2 (Ubuntu 22.04) | Android (aarch64 / armv7, API 26+) |
| :--- | :--- | :---: | :---: | :---: |
| **Host Memory Pointers** | Standard contiguous CPU memory buffer (RGB/RGBA) | ✔ Verified | ✔ Verified | ○ Supported |
| **Staging Ring Pool** | Pinned / mapped host-to-device buffer pool | ✔ Verified | ✔ Verified | ○ Supported |
| **Linux DMA-BUF** | `dma_buf_fd` (Vulkan / EGL / DRM PRIME zero-copy) | ○ Supported | ◐ Fallback (Unverified) | — |
| **AHardwareBuffer** | `AHardwareBuffer*` zero-copy interop | — | — | ○ Supported |

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
    scalix::ResizeOptions options{
        .filter = scalix::Filter::Bilinear,
        .vulkan = {
            .strategy = scalix::Strategy::LodPyramid,
            .max_mip_levels = 2, // Hierarchical anti-aliased downscale
        },
    };

    // Mode A: Synchronous
    engine.resize(src, dst, options);

    // Mode B: Asynchronous Future
    auto future = engine.resize_async(src, dst, options);
    future.get(); // Wait for completion

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

ScalixResizeOptions options = {
    .filter = SCALIX_FILTER_BILINEAR,
    .vulkan = {
        .strategy = SCALIX_STRATEGY_LOD_PYRAMID,
        .max_mip_levels = 2,
    },
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

# Run multi-resolution performance benchmark
make -C examples benchmark

# Clean example build artifacts
make -C examples clean
```

#### Individual Examples:
* **`cpp_basic`**: Demonstrates synchronous, asynchronous callback, and zero-copy DMA buffer pre-allocation.
* **`cpp_jpeg`**: Loads [`assets/sample.jpg`](assets/sample.jpg) using `libjpeg-turbo`, decodes directly into memory-mapped DMA buffers, executes the Scalix pipeline (`blit`, `raster`, or `lod [max_mip_levels]`), and writes the output JPEG.
* **`cpp_benchmark`**: Micro-benchmarking multi-resolution downscaling workloads across 4K UHD, 1080p, and 720p to 320×320 tensors.

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


