#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cassert>
#include <chrono>
#include <thread>
#include <scalix/scalix.hpp>

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define SCALIX_HAS_OPENCV 1
#else
#define SCALIX_HAS_OPENCV 0
#endif

struct OpencvInterpolationBenchmark {
    std::string method_name;
    double avg_ms;
    double fps;
};

struct BenchmarkMetrics {
    std::string name;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t dst_w;
    uint32_t dst_h;
    size_t num_frames;
    double sync_total_ms;
    double sync_avg_ms;
    double sync_fps;
    double async_total_ms;
    double async_avg_ms;
    double async_fps;
    double speedup;
};

static BenchmarkMetrics run_resolution_benchmark(
    scalix::Engine& engine,
    const std::string& name,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames = 16
) {
    size_t src_stride = src_w * 3; // RGB888 (3 bytes per pixel)
    size_t dst_stride = dst_w * 3; // RGB888 (3 bytes per pixel)
    size_t src_size = src_stride * src_h;
    size_t dst_size = dst_stride * dst_h;

    std::cout << "\n--------------------------------------------------------" << std::endl;
    std::cout << "[Benchmark: " << name << " (" << src_w << "x" << src_h << " → " << dst_w << "x" << dst_h << ", RGB888, " << num_frames << " frames)]" << std::endl;
    std::cout << "  Memory per 1 source frame: " << (src_size / (1024.0 * 1024.0)) << " MB (Total: "
              << ((src_size * num_frames) / (1024.0 * 1024.0)) << " MB)" << std::endl;

    std::vector<std::vector<uint8_t>> src_buffers(num_frames, std::vector<uint8_t>(src_size));
    std::vector<std::vector<uint8_t>> dst_sync_buffers(num_frames, std::vector<uint8_t>(dst_size, 0));
    std::vector<std::vector<uint8_t>> dst_async_buffers(num_frames, std::vector<uint8_t>(dst_size, 0));

    for (size_t i = 0; i < num_frames; ++i) {
        uint8_t pattern = static_cast<uint8_t>((i * 17 + 0x33) & 0xFF);
        std::fill(src_buffers[i].begin(), src_buffers[i].end(), pattern);
    }

    // 1. Synchronous Blocking Resize
    auto sync_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_frames; ++i) {
        scalix::ImageDesc src_desc{
            .width = src_w,
            .height = src_h,
            .stride_bytes = src_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = src_buffers[i].data(),
            .data_len = src_size,
            .dma_buf_fd = -1,
        };
        scalix::ImageDesc dst_desc{
            .width = dst_w,
            .height = dst_h,
            .stride_bytes = dst_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = dst_sync_buffers[i].data(),
            .data_len = dst_size,
            .dma_buf_fd = -1,
        };
        engine.resize(src_desc, dst_desc, scalix::Filter::Bilinear);
    }
    auto sync_end = std::chrono::high_resolution_clock::now();
    double sync_total_ms = std::chrono::duration<double, std::milli>(sync_end - sync_start).count();
    double sync_avg_ms = sync_total_ms / num_frames;
    double sync_fps = (num_frames / sync_total_ms) * 1000.0;

    // 2. Asynchronous Pipelined Resize
    auto async_start = std::chrono::high_resolution_clock::now();
    std::vector<scalix::Task> async_tasks;
    async_tasks.reserve(num_frames);

    for (size_t i = 0; i < num_frames; ++i) {
        scalix::ImageDesc src_desc{
            .width = src_w,
            .height = src_h,
            .stride_bytes = src_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = src_buffers[i].data(),
            .data_len = src_size,
            .dma_buf_fd = -1,
        };
        scalix::ImageDesc dst_desc{
            .width = dst_w,
            .height = dst_h,
            .stride_bytes = dst_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = dst_async_buffers[i].data(),
            .data_len = dst_size,
            .dma_buf_fd = -1,
        };
        async_tasks.push_back(engine.resize_async(src_desc, dst_desc, scalix::Filter::Bilinear));
    }

    for (size_t i = 0; i < num_frames; ++i) {
        async_tasks[i].wait(0, dst_async_buffers[i].data(), dst_size);
    }
    auto async_end = std::chrono::high_resolution_clock::now();
    double async_total_ms = std::chrono::duration<double, std::milli>(async_end - async_start).count();
    double async_avg_ms = async_total_ms / num_frames;
    double async_fps = (num_frames / async_total_ms) * 1000.0;

    assert(dst_sync_buffers[0] == dst_async_buffers[0]);
    double speedup = sync_total_ms / async_total_ms;

    std::cout << "  Synchronous (Blocking):   Total = " << sync_total_ms << " ms, Avg = " << sync_avg_ms << " ms, FPS = " << sync_fps << std::endl;
    std::cout << "  Asynchronous (Pipelined): Total = " << async_total_ms << " ms, Avg = " << async_avg_ms << " ms, FPS = " << async_fps << std::endl;
    std::cout << "  → Efficiency Gain / Speedup: " << speedup << "x" << std::endl;

    return BenchmarkMetrics{
        .name = name,
        .src_w = src_w,
        .src_h = src_h,
        .dst_w = dst_w,
        .dst_h = dst_h,
        .num_frames = num_frames,
        .sync_total_ms = sync_total_ms,
        .sync_avg_ms = sync_avg_ms,
        .sync_fps = sync_fps,
        .async_total_ms = async_total_ms,
        .async_avg_ms = async_avg_ms,
        .async_fps = async_fps,
        .speedup = speedup,
    };
}

int main() {
    std::cout << "========================================================" << std::endl;
    std::cout << "[Scalix Multi-Resolution Performance Benchmark]" << std::endl;
    std::cout << "Scenario: Video Frame Stream Downscaling for Neural Model Input (→ 320x320)" << std::endl;
    std::cout << "Format: Packed RGB888 (3 Channels - Standard Tensor Input)" << std::endl;
    std::cout << "Backend: Vulkan Hardware Blitter (`vkCmdBlitImage`)" << std::endl;
    std::cout << "========================================================" << std::endl;

    scalix::Engine engine(scalix::Backend::Auto);

    constexpr uint32_t dst_w = 320;
    constexpr uint32_t dst_h = 320;
    constexpr size_t NUM_FRAMES = 16;

    // A. 4K UHD (3840x2160 -> 320x320)
    auto m_4k = run_resolution_benchmark(
        engine,
        "4K UHD (3840x2160)",
        3840,
        2160,
        dst_w,
        dst_h,
        NUM_FRAMES
    );

    // B. Full HD 1080p (1920x1080 -> 320x320)
    auto m_fhd = run_resolution_benchmark(
        engine,
        "Full HD (1920x1080)",
        1920,
        1080,
        dst_w,
        dst_h,
        NUM_FRAMES
    );

    // C. HD 720p (1280x720 -> 320x320)
    auto m_hd = run_resolution_benchmark(
        engine,
        "HD 720p (1280x720)",
        1280,
        720,
        dst_w,
        dst_h,
        NUM_FRAMES
    );

    // Summary Comparison Table
    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "                    MULTI-RESOLUTION BENCHMARK COMPARISON SUMMARY" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << std::left << std::setw(22) << "Resolution"
              << std::setw(18) << "Target"
              << std::setw(16) << "Sync Latency"
              << std::setw(14) << "Sync FPS"
              << std::setw(16) << "Async Latency"
              << std::setw(14) << "Async FPS"
              << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    auto print_row = [](const BenchmarkMetrics& m) {
        std::cout << std::left << std::setw(22) << m.name
                  << std::setw(18) << ("→ " + std::to_string(m.dst_w) + "x" + std::to_string(m.dst_h))
                  << std::setw(16) << (std::to_string(m.sync_avg_ms).substr(0, 6) + " ms")
                  << std::setw(14) << (std::to_string(m.sync_fps).substr(0, 6) + " FPS")
                  << std::setw(16) << (std::to_string(m.async_avg_ms).substr(0, 6) + " ms")
                  << std::setw(14) << (std::to_string(m.async_fps).substr(0, 6) + " FPS")
                  << (std::to_string(m.speedup).substr(0, 5) + "x") << std::endl;
    };

    print_row(m_4k);
    print_row(m_fhd);
    print_row(m_hd);
    std::cout << "==========================================================================================" << std::endl;

    // Detailed Stage-by-Stage Latency Breakdown Measurement (4K UHD)
    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "          DETAILED STAGE-BY-STAGE LATENCY BREAKDOWN MEASUREMENT (4K UHD)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << "Evaluating 4K UHD (3840x2160 → 320x320) across pipeline stages:\n" << std::endl;

    size_t rgb_src_size = 3840 * 2160 * 3;
    size_t rgb_dst_size = 320 * 320 * 3;
    std::vector<uint8_t> rgb_src(rgb_src_size, 0x55);
    std::vector<uint8_t> rgb_dst(rgb_dst_size, 0x00);

    scalix::ImageDesc rgb_src_desc{
        .width = 3840,
        .height = 2160,
        .stride_bytes = 3840 * 3,
        .format = scalix::PixelFormat::Rgb888,
        .host_ptr = rgb_src.data(),
        .data_len = rgb_src_size,
        .dma_buf_fd = -1,
    };
    scalix::ImageDesc rgb_dst_desc{
        .width = 320,
        .height = 320,
        .stride_bytes = 320 * 3,
        .format = scalix::PixelFormat::Rgb888,
        .host_ptr = rgb_dst.data(),
        .data_len = rgb_dst_size,
        .dma_buf_fd = -1,
    };

    constexpr size_t PROFILE_ROUNDS = 10;

    // Enable pluggable profiler to capture GPU hardware timestamps and stage breakdowns
    engine.set_profiling(true);
    auto t_rgb_start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < PROFILE_ROUNDS; ++r) {
        engine.resize(rgb_src_desc, rgb_dst_desc, scalix::Filter::Bilinear);
    }
    auto t_rgb_end = std::chrono::high_resolution_clock::now();
    double total_rgb_pipeline_ms = std::chrono::duration<double, std::milli>(t_rgb_end - t_rgb_start).count() / PROFILE_ROUNDS;

    auto last_prof = engine.last_profile();
    engine.set_profiling(false); // Reset to zero-overhead mode

    double cpu_unpack_ms = last_prof ? last_prof->host_unpack_ms : 0.0;
    double gpu_upload_ms = last_prof ? last_prof->gpu_upload_ms : 0.0;
    double gpu_pure_blit_ms = last_prof ? last_prof->gpu_pure_blit_ms : 0.0;
    double gpu_download_ms = last_prof ? last_prof->gpu_download_ms : 0.0;
    double cpu_repack_ms = last_prof ? last_prof->host_repack_ms : 0.0;
    double driver_sync_ms = last_prof ? last_prof->driver_sync_ms : 0.0;

    std::cout << std::left << std::setw(42) << "Pipeline Stage"
              << std::setw(16) << "Latency (ms)"
              << std::setw(14) << "% of Total"
              << "Bottleneck Source" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    auto print_stage = [total_rgb_pipeline_ms](const std::string& name, double ms, const std::string& source) {
        double pct = (ms / total_rgb_pipeline_ms) * 100.0;
        std::cout << std::left << std::setw(42) << name
                  << std::setw(16) << (std::to_string(ms).substr(0, 6) + " ms")
                  << std::setw(14) << (std::to_string(pct).substr(0, 5) + " %")
                  << source << std::endl;
    };

    print_stage("1. CPU Host RGB888 Unpack -> Staging", cpu_unpack_ms, "CPU Memory Bus (RGB->RGBA expansion)");
    print_stage("2. GPU Staging -> VRAM Image Upload", gpu_upload_ms, "PCIe / Vulkan Buffer-to-Image Copy");
    print_stage("3. Pure GPU Silicon Blit (vkCmdBlitImage)", gpu_pure_blit_ms, "GPU Hardware Blitter Units (VkQueryPool)");
    print_stage("4. GPU VRAM -> Staging Image Download", gpu_download_ms, "Vulkan Image-to-Buffer Copy");
    print_stage("5. CPU Staging Readback -> RGB888 Repack", cpu_repack_ms, "CPU Memory (RGBA->RGB packing)");
    print_stage("6. Driver Recording & Queue Synchronization", driver_sync_ms, "Vulkan Driver & vkQueueWaitIdle");
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(42) << "Total Measured Host-Memory Latency"
              << std::setw(16) << (std::to_string(total_rgb_pipeline_ms).substr(0, 6) + " ms")
              << std::setw(14) << "100.0 %"
              << "End-to-End Frame Time (" + std::to_string(1000.0 / total_rgb_pipeline_ms).substr(0, 5) + " FPS)" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  → Note: Zero-Copy DMA (`VK_KHR_external_memory_fd`) eliminates Stages 1, 2, 4, 5 entirely," << std::endl;
    std::cout << "          reducing latency to pure GPU execution (~" << (gpu_pure_blit_ms + driver_sync_ms) << " ms, >"
              << (driver_sync_ms + gpu_pure_blit_ms > 0 ? static_cast<int>(1000.0 / (driver_sync_ms + gpu_pure_blit_ms)) : 400) << " FPS)!" << std::endl;
    std::cout << "==========================================================================================" << std::endl;

#if SCALIX_HAS_OPENCV
    // Comparative Benchmark: Scalix vs OpenCV across different interpolation methods
    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << "    OPENCV vs SCALIX EXECUTION TIME COMPARISON (4K UHD 3840x2160 → 320x320, RGB888)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;

    struct InterpTest {
        std::string name;
        int cv_interp;
        scalix::Filter scalix_filter;
    };

    const std::vector<InterpTest> interp_tests = {
        {"Nearest Neighbor", cv::INTER_NEAREST, scalix::Filter::Nearest},
        {"Bilinear", cv::INTER_LINEAR, scalix::Filter::Bilinear},
        {"Bicubic", cv::INTER_CUBIC, scalix::Filter::Bicubic},
        {"Area / Box Filter", cv::INTER_AREA, scalix::Filter::Area},
        {"Lanczos", cv::INTER_LANCZOS4, scalix::Filter::Lanczos3},
    };

    cv::Mat cv_src(2160, 3840, CV_8UC3);
    cv_src.setTo(cv::Scalar(0x33, 0x66, 0x99));
    cv::Mat cv_dst(320, 320, CV_8UC3);

    // Warm-up OpenCV
    cv::resize(cv_src, cv_dst, cv::Size(320, 320), 0, 0, cv::INTER_LINEAR);

    constexpr size_t CV_ROUNDS = 10;

    std::cout << std::left << std::setw(20) << "Interpolation"
              << std::setw(18) << "OpenCV Latency"
              << std::setw(16) << "OpenCV FPS"
              << std::setw(18) << "Scalix (Async)"
              << std::setw(16) << "Scalix FPS"
              << "Speedup" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& test : interp_tests) {
        // Benchmark OpenCV
        auto cv_start = std::chrono::high_resolution_clock::now();
        for (size_t r = 0; r < CV_ROUNDS; ++r) {
            cv::resize(cv_src, cv_dst, cv::Size(320, 320), 0, 0, test.cv_interp);
        }
        auto cv_end = std::chrono::high_resolution_clock::now();
        double cv_total_ms = std::chrono::duration<double, std::milli>(cv_end - cv_start).count();
        double cv_avg_ms = cv_total_ms / CV_ROUNDS;
        double cv_fps = (CV_ROUNDS / cv_total_ms) * 1000.0;

        // Benchmark Scalix (Async Pipelined)
        auto sc_start = std::chrono::high_resolution_clock::now();
        std::vector<scalix::Task> tasks;
        tasks.reserve(CV_ROUNDS);
        for (size_t r = 0; r < CV_ROUNDS; ++r) {
            tasks.push_back(engine.resize_async(rgb_src_desc, rgb_dst_desc, test.scalix_filter));
        }
        for (size_t r = 0; r < CV_ROUNDS; ++r) {
            tasks[r].wait(0, rgb_dst.data(), rgb_dst.size());
        }
        auto sc_end = std::chrono::high_resolution_clock::now();
        double sc_total_ms = std::chrono::duration<double, std::milli>(sc_end - sc_start).count();
        double sc_avg_ms = sc_total_ms / CV_ROUNDS;
        double sc_fps = (CV_ROUNDS / sc_total_ms) * 1000.0;

        double speedup = cv_avg_ms / sc_avg_ms;

        std::cout << std::left << std::setw(20) << test.name
                  << std::setw(18) << (std::to_string(cv_avg_ms).substr(0, 6) + " ms")
                  << std::setw(16) << (std::to_string(cv_fps).substr(0, 6) + " FPS")
                  << std::setw(18) << (std::to_string(sc_avg_ms).substr(0, 6) + " ms")
                  << std::setw(16) << (std::to_string(sc_fps).substr(0, 6) + " FPS")
                  << (std::to_string(speedup).substr(0, 5) + "x") << std::endl;
    }
    std::cout << "==========================================================================================" << std::endl;
#else
    std::cout << "\n[Note: OpenCV headers not found during compilation. Install libopencv-dev to enable comparison table.]" << std::endl;
#endif

    std::cout << "\n[Multi-resolution benchmark & hardware breakdown finished successfully!]" << std::endl;
    return 0;
}
