# Scalix

**Scalix** is an extensible, high-performance, hardware-accelerated image scaling and resampling engine engineered for modern **Linux** and **Android** platforms.

Designed with a **headless-first and offscreen-first** architecture, Scalix provides a unified interface across **Vulkan Compute**, **OpenGL / GLES (EGL Headless)**, **NPU / Neural Accelerators**, **Dedicated 2D HW Engines (V4L2 M2M / DRM)**, and **Vectorized CPU SIMD (AVX-512 / AVX2 / Neon)**.

---

## Key Features

* **Headless & Offscreen Native:** No display server (X11 / Wayland) required; operates cleanly in background workers, cloud servers, and embedded pipelines.
* **Flexible Execution Models:**
  * **Synchronous (Blocking):** Direct execution for CLI tools and deterministic pipelines.
  * **Asynchronous (Future / Task):** Non-blocking polling and timeout waits with hardware timeline semaphores.
  * **Callback-Driven:** Event-driven frame completion callbacks for streaming, camera, and UI pipelines.
* **In-Process Context Worker:** Embedded worker thread pool managing accelerator context affinity (single-threaded EGL / Vulkan queue ownership) and pipelining memory staging.
* **Zero-Copy Memory Subsystem:** First-class support for Linux **DMA-BUF** and Android **AHardwareBuffer** across GPU, 2D hardware blitters, and V4L2.
* **Multi-Language APIs:** Core engine with stable **C ABI** (`libscalix.so` / `scalix.h`), idiomatic **C++20** wrapper (`scalix.hpp`), and native **Rust** crate.
* **Comprehensive Filter Suite:** Nearest Neighbor, Bilinear, Bicubic (Catmull-Rom), Lanczos2/Lanczos3, Area Averaging, and pluggable AI Super-Resolution.

---

## Compatibility & Verification Matrix

This matrix tracks the hardware backends, execution paradigms, and platform capabilities supported by Scalix, along with their active verification status.

### Legend
* 🟢 **Verified & Tested:** Fully implemented and validated with automated test suite.
* 🟡 **In Progress / Scaffolded:** Core interface or backend under active implementation.
* ⚪ **Planned / Unverified:** Supported by architectural specification, pending implementation and test verification.
* ⏸️ **Deferred:** Planned for future milestone (e.g. standalone daemon service).

---

### 1. Hardware Backends & Accelerators

| Backend Provider | Subsystem / API | Host / Silicon Target | Priority | WSL2 Dev Host | Linux (x86_64) | Linux (ARM64) | Android (NDK) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vulkan** | Compute Shader / Offscreen | Modern GPU (AMD / NVIDIA / Intel / Adreno / Mali) | **P0** | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ *Unverified* |
| **OpenGL / GLES** | EGL Headless / FBO / CS | GLES 3.1+ / GL 4.3+ | **P1** | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ *Unverified* |
| **CPU SIMD** | AVX-512 / AVX2 / FMA | x86_64 (Zen 4/5, Intel Core) | **P3** | ⚪ Supported | ⚪ Supported | N/A | N/A | ⚪ *Unverified* |
| **CPU SIMD** | ARM Neon / FP16 | aarch64 / armv7 | **P3** | ⚪ Cross-compile | N/A | ⚪ Supported | ⚪ Supported | ⚪ *Unverified* |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Rockchip RGA, NXP PXP, Allwinner G2D | **P2** | ⚪ Mock / Loopback | ⚪ Hardware Req. | ⚪ Supported | N/A | ⚪ *Unverified* |
| **NPU / AI Engine** | NNAPI / QNN / OpenVINO | Qualcomm HTP, Intel NPU, MediaTek APU | **P2** | ⚪ Mock / CPU | ⚪ OpenVINO | ⚪ QNN/NPU | ⚪ NNAPI/QNN | ⚪ *Unverified* |

---

### 2. Execution Paradigms

| Execution Mode | Description | Rust Core | C ABI | C++20 API | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Synchronous (`sync`)** | Blocking call until completion or timeout | ⚪ | ⚪ | ⚪ | ⚪ *Unverified* |
| **Asynchronous (`async`)** | Returns `TaskHandle` / `std::future` / Rust `Future` | ⚪ | ⚪ | ⚪ | ⚪ *Unverified* |
| **Callback (`callback`)** | Dispatches completion function on worker thread | ⚪ | ⚪ | ⚪ | ⚪ *Unverified* |

---

### 3. Memory & Zero-Copy Subsystems

| Feature | Interface / Handle | Linux x86_64 / WSL2 | Linux ARM64 | Android (API 26+) | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Host Memory Pointers** | Standard contiguous CPU memory buffer | ⚪ | ⚪ | ⚪ | ⚪ *Unverified* |
| **Staging Ring Pool** | Pinned / mapped host-to-device buffer pool | ⚪ | ⚪ | ⚪ | ⚪ *Unverified* |
| **Linux DMA-BUF** | `dma_buf_fd` (Vulkan / EGL / V4L2 zero-copy) | ⚪ | ⚪ | N/A | ⚪ *Unverified* |
| **AHardwareBuffer** | `AHardwareBuffer*` zero-copy interop | N/A | N/A | ⚪ | ⚪ *Unverified* |

---

## Quick Start & API Preview

### C++20 API (`<scalix/scalix.hpp>`)

```cpp
#include <scalix/scalix.hpp>

int main() {
    // 1. Initialize Scalix Engine (Vulkan with OpenGL/CPU fallback)
    scalix::Engine engine(scalix::Backend::Auto);

    scalix::ImageDesc src{
        .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
        .format = scalix::PixelFormat::RGBA8888, .host_ptr = src_data
    };

    scalix::ImageDesc dst{
        .width = 1920, .height = 1080, .stride_bytes = 1920 * 4,
        .format = scalix::PixelFormat::RGBA8888, .host_ptr = dst_data
    };

    // Mode A: Synchronous
    engine.resize(src, dst, scalix::Filter::Lanczos3);

    // Mode B: Asynchronous Future
    auto future = engine.resize_async(src, dst, scalix::Filter::Lanczos3);
    future.get(); // Wait for completion

    // Mode C: Callback-driven
    engine.resize_callback(src, dst, scalix::Filter::Lanczos3, [](int status) {
        // Handle completion
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
ScalixImageDesc dst = { .width = 1920, .height = 1080, .stride_bytes = 1920 * 4,
                        .format = SCALIX_FORMAT_RGBA8888, .host_ptr = dst_ptr, .dma_buf_fd = -1 };

// Synchronous resize
scalix_resize_sync(engine, &src, &dst, SCALIX_FILTER_BICUBIC);

scalix_engine_destroy(engine);
```

---

## Building & Testing

### Prerequisites
* **Rust Toolchain:** `rustc` & `cargo` (1.70+ recommended)
* **C/C++ Toolchain:** `g++` or `clang++` supporting C++20
* **Image Codec Libraries:** `libjpeg-dev` / `libjpeg-turbo8-dev` (for JPEG I/O examples)
* **GPU Backend Libraries:**
  * **Vulkan (P0):** `libvulkan-dev`, `vulkan-tools`, `mesa-vulkan-drivers`
  * **OpenGL / GLES (P1):** `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libgl1-mesa-dev`

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

# Clean example build artifacts
make -C examples clean
```

#### Individual Examples:
* **`cpp_basic`**: Demonstrates synchronous, asynchronous callback, and zero-copy DMA buffer pre-allocation.
* **`cpp_jpeg`**: Loads [`assets/sample.jpg`](assets/sample.jpg) using `libjpeg-turbo`, decodes directly into memory-mapped DMA buffers, executes the Scalix pipeline, and writes the output JPEG (`/tmp/output_sample.jpg`).

---

## Zero-Copy DMA Subsystem by Platform

Scalix provides unified zero-copy DMA buffer allocation across supported target environments:

### 1. Bare-Metal Linux (x86_64 / aarch64, Kernel 5.6+)
* **Allocators:** Uses **DMA-Heap** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`) with fallback to **DRM Render Node Dumb Buffers** (`/dev/dri/renderD128` via GEM PRIME).
* **Permissions:** Ensure the executing user is added to `render` and `video` groups:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### 2. WSL2 (Windows Subsystem for Linux 2)
* **GPU Acceleration:** Fully supported via Microsoft DirectX bridge (`/dev/dxg`) and Mesa Vulkan/D3D12.
* **DMA Allocation Behavior:** Stock WSL2 kernels do not enable `/dev/dma_heap` by default. Scalix's runtime probing safely detects this and automatically falls back to contiguous host memory, ensuring seamless execution on WSL2 dev environments.

### 3. Android (API Level 26+, `aarch64` only)
* **Allocator:** Native **`AHardwareBuffer`** (`AHardwareBuffer_allocate`, `AHardwareBuffer_lock`).
* **Platform Gating:** Exclusively compiled for `aarch64-linux-android` (`-landroid`). Android symbols are never linked or exposed on Linux builds.

---

## Project Architecture

For deep-dive architectural specifications, backend traits, pipeline hierarchies, and threading topology, refer to [REFACTOR.md](file:///home/duty/workspace/scalix/REFACTOR.md).

---

## License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](file:///home/duty/workspace/scalix/LICENSE) file for details.


