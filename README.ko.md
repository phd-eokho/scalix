# Scalix

[English](README.md) | [한국어](README.ko.md)

**Scalix**는 최신 **Linux (x86_64)** 및 **Android (aarch64 / armv7)** 플랫폼을 위해 설계된 고성능 하드웨어 가속 이미지 스케일링 및 리샘플링 엔진입니다.

**Headless 및 Offscreen 전용** 아키텍처로 설계되어 디스플레이 서버 없이도 **Vulkan Compute**, **OpenGL / GLES (EGL Headless)**, **NPU / Neural Accelerators**, **2D HW Engines (V4L2 M2M / DRM)** 전반에 걸쳐 통합된 인터페이스를 제공합니다. CPU Fallback 및 Host SIMD 연산은 OpenCV 등 서드파티 이미지 처리 라이브러리로 위임됩니다.

---

## 주요 기능

* **Headless & Offscreen Native:** X11이나 Wayland 같은 디스플레이 서버 없이 백그라운드 워커, 클라우드 서버, 임베디드 파이프라인에서 순수 Offscreen GPU 워크로드를 수행합니다.
* **Vulkan Offscreen 렌더링 전략:**
  * **Hardware Blit (`Blit`):** Shader 오버헤드 없이 최대 처리량을 제공하는 고정 기능(Fixed-Function) 2D Blitter.
  * **Offscreen Raster Graphics (`Raster`):** Vertex/Fragment Shader 및 하드웨어 Bilinear/Trilinear Sampler를 사용하는 완전한 그래픽스 파이프라인.
  * **Hierarchical LoD Pyramid (`LodPyramid`):** 극단적인 다운스케일(>4×) 환경에서 계단 현상(Aliasing)과 모아레(Moiré) 왜곡을 제거하기 위한 2×2 박스 필터링 기반 Multi-Pass 다운스케일러 (조절 가능한 `max_mip_levels` 지원).
* **GPU Compute Shader 가속:** GPU 상에서 직접 Packed RGB888 Unpack 및 Repack 패스를 수행하는 전용 컴퓨트 파이프라인(`VulkanRgbCompute`)을 제공하여 CPU 메모리 변환 병목을 제거합니다.
* **Ring-Buffered Staging Allocator:** 슬롯별 `VkFence` 동기화와 히스테리시스 기반 메모리 자동 축소 정책(<50% 용량 기준)을 갖춘 Triple Buffering Staging 링(`VulkanStagingRing`, 3 슬롯)을 통해 CPU/GPU 간 원활한 파이프라이닝을 지원합니다.
* **유연한 실행 모델:**
  * **동기식 (Synchronous / Blocking):** CLI 도구 및 결정론적 파이프라인을 위한 직접 블로킹 실행.
  * **비동기식 (Zero-Copy Task):** 복사 없는 디스크립터 디스패치(`resize_async_raw`) 및 논블로킹 폴링/타임아웃 대기.
  * **콜백 기반 (Callback-Driven):** 스트리밍, 카메라, UI 파이프라인을 위한 이벤트 기반 프레임 완료 콜백.
* **엄격한 64바이트 메모리 정렬:** 모든 디스크립터에 64바이트 버퍼 정렬(`SCALIX_REQUIRED_ALIGNMENT_BYTES = 64`)을 강제하여 AVX-512 / ARM Neon SIMD 벡터화 및 DMA-BUF 하드웨어 호환성을 보장합니다.
* **Zero-Copy 메모리 서브시스템:** GPU, 2D 하드웨어 Blitter, V4L2 간 Linux **DMA-BUF** 및 Android **AHardwareBuffer** 네이티브 지원.
* **다국어 API 바인딩:** 안정적인 **C ABI** (`libscalix.so` / `scalix.h`), 현대적인 **C++20** 래퍼 (`scalix.hpp`), 네이티브 **Rust** 크레이트 제공.
* **포괄적인 필터 제품군:** Nearest Neighbor, Bilinear, Bicubic 및 계층형 Mipchain 다운스케일링.

---

## 호환성 및 검증 매트릭스 (Compatibility Matrix)

이 매트릭스는 Scalix가 지원하는 하드웨어 백엔드, 메모리 서브시스템, 플랫폼 기능 및 현재 검증 상태를 나타냅니다.

#### 범례
* `✔` **Verified & Tested:** 완벽히 구현되었으며 자동화된 테스트 스위트 및 벤치마크로 검증 완료.
* `◐` **Compiled / In Progress:** CI 상에서 크로스 컴파일 빌드가 완료되었거나 핵심 인터페이스 구현 진행 중 (실제 하드웨어 런타임 테스트 대기).
* `○` **Planned / Unverified:** 아키텍처 규격상 지원 예정이며 구현 및 검증 대기 중.
* `—` **Deferred / N/A:** 향후 마일스톤으로 연기되었거나 대상 플랫폼에 해당하지 않음.

---

### 1. 하드웨어 백엔드 및 가속기

| Backend Provider | Subsystem / API | Host / Silicon Target | WSL2 Dev Host | Linux (x86_64) | Android (aarch64 / armv7) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Vulkan Offscreen** | Graphics (`Blit`, `Raster`, `LodPyramid`, `Compute`) | Modern GPU (AMD / NVIDIA / Intel / Mesa Lavapipe) | ✔ Verified | ✔ Verified | ◐ Compiled (미검증) |
| **OpenGL / GLES** | EGL Headless / FBO / CS (`Blit`, `Raster`, `LodPyramid`, `Compute`) | GLES 3.1+ / GL 4.3+ / Mesa | ✔ Verified | ✔ Verified | ◐ Compiled (미검증) |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Rockchip RGA, NXP PXP, Allwinner G2D | ○ Mock / Loopback | ○ Hardware Req. | — |
| **NPU / AI Engine** | NNAPI / QNN / OpenVINO | Qualcomm HTP, Intel NPU, MediaTek APU | ○ Mock / CPU | ○ OpenVINO | ○ QNN / NNAPI |

---

### 2. 메모리 및 Zero-Copy 서브시스템

| 기능 | 인터페이스 / 핸들 | Bare-Metal Linux (x86_64) | WSL2 (Ubuntu 22.04) | Android (aarch64 / armv7, API 26+) |
| :--- | :--- | :---: | :---: | :---: |
| **64바이트 정렬 Host Memory** | 64바이트 정렬된 연속형 CPU 메모리 버퍼 (RGB/RGBA) | ✔ Verified | ✔ Verified | ◐ Compiled (미검증) |
| **Triple-Buffered Staging Ring** | 히스테리시스 축소 정책이 적용된 3-슬롯 Staging 버퍼 링 | ✔ Verified | ✔ Verified | ◐ Compiled (미검증) |
| **Linux DMA-BUF** | `dma_buf_fd` (Vulkan / EGL / DRM PRIME Zero-Copy) | ○ Supported | ◐ Fallback | — |
| **AHardwareBuffer** | `AHardwareBuffer*` Zero-Copy 연동 | — | — | ◐ Compiled (미검증) |

> [!NOTE]
> **현재 검증 상태 및 타겟 플랫폼 현황**
> - **Linux (x86_64):** 실제 NVIDIA GPU(Vulkan 드라이버) 및 CI 파이프라인(Mesa Lavapipe Vulkan 소프트웨어 래스터라이저)에서 검증되었습니다.
> - **WSL2 (Windows Subsystem for Linux 2):** NVIDIA GPU 환경의 `/dev/dxg` 브리지를 통한 Vulkan 오프스크린 렌더링이 검증되었습니다. Linux `dma-buf`는 64바이트 정렬 Host Staging 메모리로 자동 Fallback되어 원활히 동작합니다.
> - **Android (aarch64 / armv7):** CI 상에서 Android NDK 크로스 컴파일(`cargo-ndk`) 및 동적 라이브러리 빌드가 검증되었습니다. 단, 실제 Android 기기나 에뮬레이터 상에서의 런타임 GPU 실행, Vulkan 드라이버 구동, `AHardwareBuffer` Zero-Copy DMA 동작은 **아직 검증되지 않았습니다 (Unverified)**.

---

### 3. 픽셀 포맷 및 필터 전략 실행 매트릭스 (Vulkan Backend)

| 픽셀 포맷 (In / Out) | `Nearest` | `Bilinear` | `Bicubic` | `Lanczos3` | `Area` | `LodPyramid` (축소) | 기본 `Auto` 전략 매핑 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RGB888` / `BGR888` (24-bit)** | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Raster` / `Compute` (✔) | **`Compute`** (Fused 24-bit 전용 패스) |
| **`RGBA8888` / `BGRA8888` (32-bit)** | `Blit` / `Compute` (✔) | `Blit` / `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `Compute` (✔) | `LodPyramid` (✔) | **`Blit`** (고속 패스) / **`Compute`** (고차 필터) |
| **`R8` / `RG88` (Single/Dual Ch)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`RGBA16F` / `RGBA32F` (HDR/Float)** | `Blit` / `Raster` (✔) | `Blit` / `Raster` (✔) | `Raster` (◐) | `Raster` (◐) | `Raster` (◐) | `Raster` (✔) | **`Blit`** |
| **`NV12` / `YUV420p` (Semi/Planar)** | `Raster` / `Blit` (◐) | `Raster` / `Blit` (◐) | — | — | — | — | **`Raster`** (Y/UV 평면 분할 패스) |

> [!NOTE]
> **필터 알고리즘 레퍼런스 구현 (Reference Implementation)**
> - **Bicubic (`FilterMode::Bicubic`):** $4 \times 4$ 탭 윈도우 기반 2D 분리형 Catmull-Rom 3차 스플라인 보간 ($a = -0.5$) 적용 ([Keys, 1981](https://doi.org/10.1109/TASSP.1981.1163711)).
> - **Lanczos3 (`FilterMode::Lanczos3`):** 밝기 왜곡을 방지하기 위한 동적 가중치 정규화가 적용된 3-lobe sinc 윈도우 sinc 필터 ($a = 3$, $L(x) = \text{sinc}(x)\text{sinc}(x/3)$) 기반 $6 \times 6$ 탭 윈도우 보간 ([Lanczos, 1956](https://archive.org/details/appliedanalysis0000corn); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **Area (`FilterMode::Area`):** 소스 텍셀에 대한 서브픽셀 2D 경계 영역 오버랩 가중치를 정확히 적분하는 픽셀 영역 관계(Box Average) 보간 적용 (에너지 보존 및 에일리어싱 방지) ([Crow, 1984](https://doi.org/10.1145/964965.808599); [Turkowski, 1990](https://dl.acm.org/doi/10.5555/90767.90797)).
> - **전략 자동 라우팅 (`VulkanStrategy::Auto`):** CPU 호스트 포맷 변환 병목을 방지하기 위해 24-bit 패킹 포맷(`RGB888` / `BGR888`)은 Fused GPU `Compute`로 직행합니다. 32-bit `RGBA8888`의 `Nearest` 및 `Bilinear`는 최대 필레이트 처리를 위해 하드웨어 `Blit` 명령을 우선 사용하며, `Bicubic`, `Lanczos3`, `Area` 고차 필터는 GPU `Compute` 파이프라인으로 디스패치됩니다.

---

## 빠른 시작 및 API 미리보기

### C++20 API (`<scalix/scalix.hpp>`)

```cpp
#include <scalix/scalix.hpp>

int main() {
    // 1. Scalix Engine 초기화
    scalix::Engine engine(scalix::Backend::Auto);
    engine.set_profiling(true); // 선택 사항: 하드웨어 레이턴시 프로파일링 활성화

    scalix::ImageDesc src{
        .width = 3840, .height = 2160, .stride_bytes = 3840 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = src_data
    };

    scalix::ImageDesc dst{
        .width = 320, .height = 320, .stride_bytes = 320 * 4,
        .format = scalix::PixelFormat::Rgba8888, .host_ptr = dst_data
    };

    // 동적 리사이즈 옵션 설정 (Strategy: Blit, Raster, LodPyramid)
    const scalix::ResizeOptions options = scalix::ResizeOptions::with_vulkan(
        scalix::Filter::Bilinear,
        scalix::Strategy::LodPyramid,
        2 // 계층적 Anti-Aliasing 다운스케일링
    );

    // Mode A: 동기식 실행
    engine.resize(src, dst, options);

    // Mode B: 비동기 Task 실행
    auto task = engine.resize_async(src, dst, options);
    task.wait(); // 완료 대기

    // Mode C: 콜백 기반 실행
    engine.resize_callback(src, dst, options, [](int status) {
        // 워커 스레드 풀에서 완료 처리
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

// 동적 옵션을 적용한 동기식 리사이즈
scalix_resize_sync_with_options(engine, &src, &dst, &options);

scalix_engine_destroy(engine);
```

---

## 빌드 및 테스트

### 사전 요구사항
* **Rust 툴체인:** `rustc` 및 `cargo` (1.70+ 권장)
* **C/C++ 툴체인:** C++20을 지원하는 `g++` 또는 `clang++`
* **이미지 코덱 라이브러리:** `libjpeg-dev` / `libjpeg-turbo8-dev` (JPEG I/O 예제용)
* **벤치마크 및 비교 라이브러리:** `libopencv-dev` (OpenCV CPU/GPU 연산과 Scalix 하드웨어 파이프라인의 실행 시간 비교 벤치마크에 필요)
* **GPU 백엔드 라이브러리:**
  * **Vulkan (P0):** `libvulkan-dev`, `vulkan-tools`, `mesa-vulkan-drivers`
  * **OpenGL / GLES (P1):** `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libgl1-mesa-dev`
* **Android 크로스 컴파일 (선택 사항):** Android NDK (r27+ 권장, API 레벨 26+) 및 `cargo-ndk`

### 1. Rust 코어 및 C ABI 라이브러리 빌드
정적/동적 라이브러리(`libscalix.so` / `libscalix.a`)를 빌드합니다:
```bash
# 디버그 빌드
cargo build

# 최적화 릴리즈 빌드 (target/release/libscalix.so 생성)
cargo build --release
```

### 2. 테스트 스위트 실행
모든 단위 테스트, 통합 테스트 및 멀티스레드 콜백 테스트를 실행합니다:
```bash
cargo test
```

### 3. C++ 예제 빌드 및 실행
예제는 [`examples/Makefile`](examples/Makefile)을 통해 관리됩니다. 작업 공간 루트에서 직접 빌드하고 실행할 수 있습니다:

```bash
# 전체 예제 빌드 (필요시 Cargo 작업 공간 자동 빌드)
make -C examples

# 전체 예제 실행 (libjpeg-turbo 기반 sample.jpg JPEG 처리 포함)
make -C examples run

# 모든 성능 벤치마크 실행 (플랫폼에서 지원하는 모든 백엔드: Vulkan & OpenGL/EGL)
make -C examples benchmark

# 다중 해상도 성능 벤치마크 실행 (Vulkan)
make -C examples benchmark_vulkan

# 다중 해상도 성능 벤치마크 실행 (OpenGL/EGL)
make -C examples benchmark_gl

# 예제 빌드 산출물 정리
make -C examples clean
```

#### 개별 예제 안내:
* **`cpp_basic`**: 동기식, 비동기 콜백 및 Zero-Copy DMA 버퍼 사전 할당 데모.
* **`cpp_jpeg`**: `libjpeg-turbo`를 사용하여 [`assets/sample.jpg`](assets/sample.jpg)를 메모리 맵핑된 DMA 버퍼로 직접 디코딩하고, Scalix 파이프라인(`blit`, `raster`, `lod [max_mip_levels]`)을 실행한 후 결과 JPEG를 저장.
* **`cpp_benchmark`**: 4K UHD, 1080p, 720p 입력을 320×320 텐서로 다운스케일링하는 **Vulkan** 다중 해상도 마이크로 벤치마크.
* **`cpp_benchmark_gl`**: 4K UHD, 1080p, 720p 입력을 320×320 텐서로 다운스케일링하는 **OpenGL / EGL** 다중 해상도 마이크로 벤치마크.

---

## 플랫폼별 Zero-Copy DMA 서브시스템

Scalix는 지원되는 실행 환경 전반에서 통합된 Zero-Copy DMA 버퍼 할당 방식을 제공합니다:

### 1. Bare-Metal Linux (x86_64, 커널 5.6+)
* **할당자(Allocators):** **DMA-Heap** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`)을 사용하며, 부재 시 **DRM Render Node Dumb Buffers** (`/dev/dri/renderD128`, GEM PRIME 기반)로 자동 Fallback.
* **권한 설정:** 실행 사용자가 `render` 및 `video` 그룹에 추가되어 있어야 합니다:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### 2. WSL2 (Windows Subsystem for Linux 2: Ubuntu 22.04, NVIDIA GPU + AMD CPU)
* **GPU 가속:** Microsoft DirectX 브리지 (`/dev/dxg`) 및 Mesa Vulkan/D3D12를 통해 오프스크린 렌더링을 완벽 지원.
* **DMA 할당 동작 및 제약사항:** WSL2 환경에서는 Linux `dma-buf`가 **아직 검증되지 않았습니다**. 기본 WSL2 커널에는 `/dev/dma_heap`이 포함되어 있지 않으며, DRM GEM Dumb 버퍼 할당 역시 `/dev/dxg` 가상화 계층으로 인해 실패하거나 PRIME 하드웨어 내보내기가 지원되지 않는 한계가 있습니다. Scalix의 런타임 탐지 로직은 이러한 DMA 실패를 감지하여 연속형 Host 메모리 Staging 버퍼로 안전하게 자동 Fallback합니다. 실제 Zero-Copy DMA 검증은 네이티브 DRM 렌더 노드를 지원하는 Bare-Metal Linux 환경에서 진행되어야 합니다.

### 3. Android (API Level 26+, `aarch64` / `armv7`)
* **할당자(Allocator):** 네이티브 **`AHardwareBuffer`** (`AHardwareBuffer_allocate`, `AHardwareBuffer_lock`).
* **플랫폼 게이팅:** `aarch64-linux-android` 및 `armv7-linux-androideabi` (`-landroid`) 대상으로 컴파일되며 NDK r27+을 사용합니다. Android 전용 심볼은 Linux 빌드 시 절대 링크되거나 노출되지 않습니다.

---

## 라이선스 (License)

이 프로젝트는 **MIT License** 조건에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하십시오.
