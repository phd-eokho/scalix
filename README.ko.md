# Scalix

**Scalix**는 최신 **Linux** 및 **Android** 플랫폼을 위해 설계된 고성능 하드웨어 가속 이미지 스케일링 및 리샘플링 엔진입니다.

**Headless 및 Offscreen 전용** 아키텍처로 설계되어 디스플레이 서버 없이도 **Vulkan Compute**, **OpenGL / GLES (EGL Headless)**, **NPU / Neural Accelerators**, **2D HW Engines (V4L2 M2M / DRM)**, **Vectorized CPU SIMD (AVX-512 / AVX2 / Neon)** 전반에 걸쳐 통합된 인터페이스를 제공합니다.

---

## 주요 기능

* **Headless & Offscreen Native:** X11이나 Wayland 같은 디스플레이 서버 없이 백그라운드 워커, 클라우드 서버, 임베디드 파이프라인에서 순수 Offscreen GPU 워크로드를 수행합니다.
* **Vulkan Offscreen 렌더링 전략:**
  * **Hardware Blit (`Blit`):** Shader 오버헤드 없이 최대 처리량을 제공하는 고정 기능(Fixed-Function) 2D Blitter.
  * **Offscreen Raster Graphics (`Raster`):** Vertex/Fragment Shader 및 하드웨어 Bilinear/Trilinear Sampler를 사용하는 완전한 그래픽스 파이프라인.
  * **Hierarchical LoD Pyramid (`LodPyramid`):** 극단적인 다운스케일($>4\times$) 환경에서 계단 현상(Aliasing)과 모아레(Moiré) 왜곡을 제거하기 위한 $2\times 2$ 박스 필터링 기반 Multi-Pass 다운스케일러 (조절 가능한 `max_mip_levels` 지원).
* **유연한 실행 모델:**
  * **동기식 (Synchronous / Blocking):** CLI 도구 및 결정론적 파이프라인을 위한 직접 블로킹 실행.
  * **비동기식 (Asynchronous / Future / Task):** 하드웨어 Timeline Semaphore 기반의 Non-blocking 폴링 및 타임아웃 대기.
  * **콜백 기반 (Callback-Driven):** 스트리밍, 카메라, UI 파이프라인을 위한 이벤트 기반 프레임 완료 콜백.
* **In-Process Context Worker:** 가속기 Context Affinity(단일 스레드 EGL / Vulkan Queue 소유권)와 메모리 Staging 파이프라이닝을 관리하는 임베디드 워커 스레드 풀.
* **Zero-Copy 메모리 서브시스템:** GPU, 2D 하드웨어 Blitter, V4L2 간 Linux **DMA-BUF** 및 Android **AHardwareBuffer** 네이티브 지원.
* **다국어 API 바인딩:** 안정적인 **C ABI** (`libscalix.so` / `scalix.h`), 현대적인 **C++20** 래퍼 (`scalix.hpp`), 네이티브 **Rust** 크레이트 제공.
* **포괄적인 필터 제품군:** Nearest Neighbor, Bilinear, Bicubic 및 계층형 Mipchain 다운스케일링.

---

## 호환성 및 검증 매트릭스 (Compatibility Matrix)

이 매트릭스는 Scalix가 지원하는 하드웨어 백엔드, 실행 패러다임, 플랫폼 기능 및 현재 검증 상태를 나타냅니다.

### 범례
* 🟢 **Verified & Tested:** 완벽히 구현되었으며 자동화된 테스트 스위트 및 벤치마크로 검증 완료.
* 🟡 **In Progress / Scaffolded:** 핵심 인터페이스 또는 백엔드 구현 진행 중.
* ⚪ **Planned / Unverified:** 아키텍처 규격상 지원 예정이며 구현 및 검증 대기 중.
* ⏸️ **Deferred:** 향후 마일스톤(예: 독립 데몬 서비스)으로 연기됨.

---

### 1. 하드웨어 백엔드 및 가속기

| Backend Provider | Subsystem / API | Host / Silicon Target | Priority | WSL2 Dev Host | Linux (x86_64) | Linux (ARM64) | Android (NDK) | 검증 상태 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vulkan Offscreen** | Graphics (`Blit`, `Raster`, `LodPyramid`) | Modern GPU (AMD / NVIDIA / Intel / Mesa LLVMpipe) | **P0** | 🟢 Verified | 🟢 Verified | ⚪ Supported | ⚪ Supported | 🟢 **Verified** |
| **OpenGL / GLES** | EGL Headless / FBO / CS | GLES 3.1+ / GL 4.3+ | **P1** | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ Supported | ⚪ *Planned* |
| **CPU SIMD** | AVX-512 / AVX2 / FMA | x86_64 (Zen 4/5, Intel Core) | **P3** | ⚪ Supported | ⚪ Supported | N/A | N/A | ⚪ *Planned* |
| **CPU SIMD** | ARM Neon / FP16 | aarch64 / armv7 | **P3** | ⚪ Cross-compile | N/A | ⚪ Supported | ⚪ Supported | ⚪ *Planned* |
| **2D HW Blitter** | V4L2 M2M / DRM Scaler | Rockchip RGA, NXP PXP, Allwinner G2D | **P2** | ⚪ Mock / Loopback | ⚪ Hardware Req. | ⚪ Supported | N/A | ⚪ *Planned* |
| **NPU / AI Engine** | NNAPI / QNN / OpenVINO | Qualcomm HTP, Intel NPU, MediaTek APU | **P2** | ⚪ Mock / CPU | ⚪ OpenVINO | ⚪ QNN/NPU | ⚪ NNAPI/QNN | ⚪ *Planned* |

---

### 2. 실행 패러다임

| 실행 모드 | 설명 | Rust Core | C ABI | C++20 API | 검증 상태 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **동기식 (`sync`)** | GPU 작업 완료 또는 타임아웃까지 블로킹 호출 | 🟢 | 🟢 | 🟢 | 🟢 **Verified** |
| **비동기식 (`async`)** | `TaskHandle` / `std::future` / Rust `Future` 반환 | 🟢 | 🟢 | 🟢 | 🟢 **Verified** |
| **콜백 (`callback`)** | 워커 스레드 풀에서 비동기 완료 콜백 함수 디스패치 | 🟢 | 🟢 | 🟢 | 🟢 **Verified** |

---

### 3. 메모리 및 Zero-Copy 서브시스템

| 기능 | 인터페이스 / 핸들 | Linux x86_64 / WSL2 | Linux ARM64 | Android (API 26+) | 검증 상태 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Host Memory Pointers** | 표준 연속형 CPU 메모리 버퍼 (RGB/RGBA) | 🟢 Verified | ⚪ Supported | ⚪ Supported | 🟢 **Verified** |
| **Staging Ring Pool** | Pinned / Mapped Host-to-Device 버퍼 풀 | 🟢 Verified | ⚪ Supported | ⚪ Supported | 🟢 **Verified** |
| **Linux DMA-BUF** | `dma_buf_fd` (Vulkan / EGL / DRM PRIME Zero-Copy) | 🟢 Probed / Fallback | ⚪ Supported | N/A | 🟢 **Verified** |
| **AHardwareBuffer** | `AHardwareBuffer*` Zero-Copy 연동 | N/A | N/A | ⚪ | ⚪ *Planned* |

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
    scalix::ResizeOptions options{
        .filter = scalix::Filter::Bilinear,
        .vulkan = {
            .strategy = scalix::Strategy::LodPyramid,
            .max_mip_levels = 2, // 계층적 Anti-Aliasing 다운스케일링
        },
    };

    // Mode A: 동기식 실행
    engine.resize(src, dst, options);

    // Mode B: 비동기 Future 실행
    auto future = engine.resize_async(src, dst, options);
    future.get(); // 완료 대기

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

ScalixResizeOptions options = {
    .filter = SCALIX_FILTER_BILINEAR,
    .vulkan = {
        .strategy = SCALIX_STRATEGY_LOD_PYRAMID,
        .max_mip_levels = 2,
    },
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
* **GPU 백엔드 라이브러리:**
  * **Vulkan (P0):** `libvulkan-dev`, `vulkan-tools`, `mesa-vulkan-drivers`
  * **OpenGL / GLES (P1):** `libegl1-mesa-dev`, `libgles2-mesa-dev`, `libgl1-mesa-dev`

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

# 예제 빌드 산출물 정리
make -C examples clean
```

#### 개별 예제 안내:
* **`cpp_basic`**: 동기식, 비동기 콜백 및 Zero-Copy DMA 버퍼 사전 할당 데모.
* **`cpp_jpeg`**: `libjpeg-turbo`를 사용하여 [`assets/sample.jpg`](assets/sample.jpg)를 메모리 맵핑된 DMA 버퍼로 직접 디코딩하고, Scalix 파이프라인(`blit`, `raster`, `lod [max_mip_levels]`)을 실행한 후 결과 JPEG를 저장.
* **`cpp_benchmark`**: 4K UHD, 1080p, 720p 입력을 $320\times 320$ 텐서로 다운스케일링하는 다중 해상도 마이크로 벤치마크.

---

## 플랫폼별 Zero-Copy DMA 서브시스템

Scalix는 지원되는 실행 환경 전반에서 통합된 Zero-Copy DMA 버퍼 할당 방식을 제공합니다:

### 1. Bare-Metal Linux (x86_64 / aarch64, 커널 5.6+)
* **할당자(Allocators):** **DMA-Heap** (`/dev/dma_heap/system`, `/dev/dma_heap/cma`)을 사용하며, 부재 시 **DRM Render Node Dumb Buffers** (`/dev/dri/renderD128`, GEM PRIME 기반)로 자동 Fallback.
* **권한 설정:** 실행 사용자가 `render` 및 `video` 그룹에 추가되어 있어야 합니다:
  ```bash
  sudo usermod -aG render,video $USER
  ```

### 2. WSL2 (Windows Subsystem for Linux 2)
* **GPU 가속:** Microsoft DirectX 브리지 (`/dev/dxg`) 및 Mesa Vulkan/D3D12를 통해 완벽 지원.
* **DMA 할당 동작:** 기본 WSL2 커널은 `/dev/dma_heap`을 기본 활성화하지 않습니다. Scalix의 런타임 탐지 기능은 이를 안전하게 감지하고 연속형 Host 메모리로 자동 Fallback하여 WSL2 개발 환경에서도 원활하게 동작합니다.

### 3. Android (API Level 26+, `aarch64` 전용)
* **할당자(Allocator):** 네이티브 **`AHardwareBuffer`** (`AHardwareBuffer_allocate`, `AHardwareBuffer_lock`).
* **플랫폼 게이팅:** `aarch64-linux-android` (`-landroid`) 전용으로 컴파일됩니다. Android 전용 심볼은 Linux 빌드 시 절대 링크되거나 노출되지 않습니다.

---

## 라이선스 (License)

이 프로젝트는 **MIT License** 조건에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하십시오.
