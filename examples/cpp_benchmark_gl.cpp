#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <optional>
#include <filesystem>
#include <cassert>
#include <scalix/scalix.hpp>
#include "jpeg_io.hpp"
#include "benchmark_utils.hpp"

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define SCALIX_HAS_OPENCV 1
#else
#define SCALIX_HAS_OPENCV 0
#endif

using namespace std;
namespace fs = std::filesystem;

using scalix::AlignedVector;
using scalix::examples::JpegIO;
using scalix::examples::JpegHeader;
using scalix::examples::MetricStats;
using scalix::examples::ResolutionBenchmarkResult;
using scalix::examples::StrategyBenchmarkResult;
using scalix::examples::TableReporter;
using scalix::examples::PreciseTimer;
using scalix::examples::VERIFY_JPEG_QUALITY;

namespace {

constexpr string_view DEFAULT_SAMPLE_PATH   = "assets/sample.jpg";
constexpr string_view FALLBACK_SAMPLE_PATH  = "../assets/sample.jpg";

constexpr string_view PREFIX_BENCH          = "bench_gl";
constexpr string_view UNIT_MS               = " ms";
constexpr string_view UNIT_FPS              = " FPS";
constexpr string_view UNIT_X                = "x";

constexpr size_t DEFAULT_BENCH_ROUNDS       = 16;
constexpr size_t PROFILING_ROUNDS           = 5;
constexpr size_t CV_BENCH_ROUNDS            = 10;
constexpr uint32_t DOWNSCALE_RATIO          = 8;
constexpr uint32_t SYNTH_DST_DIM            = 320;

constexpr uint32_t SYNTH_4K_W               = 3840;
constexpr uint32_t SYNTH_4K_H               = 2160;
constexpr uint32_t SYNTH_FHD_W              = 1920;
constexpr uint32_t SYNTH_FHD_H              = 1080;
constexpr uint32_t SYNTH_HD_W               = 1280;
constexpr uint32_t SYNTH_HD_H               = 720;

constexpr string_view OUT_BLIT_FILE         = "/tmp/bench_gl_sample_blit.jpg";
constexpr string_view OUT_RASTER_FILE       = "/tmp/bench_gl_sample_raster.jpg";
constexpr string_view OUT_LOD_AUTO_FILE     = "/tmp/bench_gl_sample_lod_auto.jpg";
constexpr string_view OUT_LOD_2LVL_FILE     = "/tmp/bench_gl_sample_lod_2lvl.jpg";
constexpr string_view OUT_COMPUTE_FILE      = "/tmp/bench_gl_sample_compute.jpg";
constexpr string_view OUT_AREA_FILE         = "/tmp/bench_gl_sample_area.jpg";
constexpr string_view OUT_AUTO_FILE         = "/tmp/bench_gl_sample_auto.jpg";

constexpr string_view MSG_DMA_ACTIVE        = "[Hardware DMA Status: Native Zero-Copy DMA Active (DMA-Heap / DRM GEM Dumb)]";
constexpr string_view MSG_DMA_STAGING       = "[Hardware DMA Status: Host-Memory Staging Mode Active (Hardware DMA-BUF / DRM unavailable in current environment)]";

} // anonymous namespace

// ============================================================================
// Synthetic Resolution Benchmarks
// ============================================================================

static ResolutionBenchmarkResult run_resolution_benchmark(
    scalix::Engine& engine,
    string_view name,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames = DEFAULT_BENCH_ROUNDS
) {
    const size_t src_stride = static_cast<size_t>(src_w) * 3;
    const size_t dst_stride = static_cast<size_t>(dst_w) * 3;
    const size_t src_size = src_stride * src_h;
    const size_t dst_size = dst_stride * dst_h;

    cout << "\n--------------------------------------------------------" << endl;
    cout << "[Benchmark: " << name << " (" << src_w << "x" << src_h << " → " << dst_w << "x" << dst_h << ", RGB888, " << num_frames << " frames)]" << endl;
    cout << "  Memory per 1 frame: " << (src_size / (1024.0 * 1024.0)) << " MB (Total: "
         << ((src_size * num_frames) / (1024.0 * 1024.0)) << " MB)" << endl;

    vector<AlignedVector<uint8_t>> src_buffers(num_frames, AlignedVector<uint8_t>(src_size));
    vector<AlignedVector<uint8_t>> dst_sync_buffers(num_frames, AlignedVector<uint8_t>(dst_size, 0));
    vector<AlignedVector<uint8_t>> dst_async_buffers(num_frames, AlignedVector<uint8_t>(dst_size, 0));

    for (size_t i = 0; i < num_frames; ++i) {
        const uint8_t pattern = static_cast<uint8_t>((i * 17 + 0x33) & 0xFF);
        fill(src_buffers[i].begin(), src_buffers[i].end(), pattern);
    }

    // 1. Synchronous Blocking
    const auto sync_stats = PreciseTimer::measure(num_frames, [&](size_t i) {
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
    });

    // 2. Asynchronous Pipelined
    vector<scalix::Task> async_tasks;
    async_tasks.reserve(num_frames);

    auto async_stats = PreciseTimer::measure(1, [&](size_t) {
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
    });
    async_stats = MetricStats::from_duration(async_stats.total_ms, num_frames);

    assert(dst_sync_buffers[0] == dst_async_buffers[0]);
    const double speedup = sync_stats.total_ms / async_stats.total_ms;

    cout << "  Synchronous (Blocking):   Total = " << sync_stats.total_ms << " ms, Avg = " << sync_stats.avg_ms << " ms, FPS = " << sync_stats.fps << endl;
    cout << "  Asynchronous (Pipelined): Total = " << async_stats.total_ms << " ms, Avg = " << async_stats.avg_ms << " ms, FPS = " << async_stats.fps << endl;
    cout << "  → Efficiency Gain / Speedup: " << speedup << "x" << endl;

    return ResolutionBenchmarkResult{
        .name = string(name),
        .src_w = src_w,
        .src_h = src_h,
        .dst_w = dst_w,
        .dst_h = dst_h,
        .num_frames = num_frames,
        .sync = sync_stats,
        .async = async_stats,
        .speedup = speedup,
    };
}

static optional<ResolutionBenchmarkResult> run_dma_resolution_benchmark(
    scalix::Engine& engine,
    string_view name,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames = DEFAULT_BENCH_ROUNDS
) {
    cout << "\n--------------------------------------------------------" << endl;
    cout << "[Zero-Copy DMA Benchmark: " << name << " (" << src_w << "x" << src_h << " → " << dst_w << "x" << dst_h << ", " << num_frames << " frames)]" << endl;

    vector<unique_ptr<scalix::DmaBuffer>> src_buffers;
    vector<unique_ptr<scalix::DmaBuffer>> dst_sync_buffers;
    vector<unique_ptr<scalix::DmaBuffer>> dst_async_buffers;
    src_buffers.reserve(num_frames);
    dst_sync_buffers.reserve(num_frames);
    dst_async_buffers.reserve(num_frames);

    try {
        for (size_t i = 0; i < num_frames; ++i) {
            auto src_buf = make_unique<scalix::DmaBuffer>(src_w, src_h, scalix::PixelFormat::Rgba8888, scalix::AllocatorType::Auto);
            const uint8_t pattern = static_cast<uint8_t>((i * 17 + 0x33) & 0xFF);
            src_buf->with_write([pattern](uint8_t* ptr, size_t size) {
                if (ptr && size > 0) memset(ptr, pattern, size);
            });
            src_buffers.push_back(move(src_buf));
            dst_sync_buffers.push_back(make_unique<scalix::DmaBuffer>(dst_w, dst_h, scalix::PixelFormat::Rgba8888, scalix::AllocatorType::Auto));
            dst_async_buffers.push_back(make_unique<scalix::DmaBuffer>(dst_w, dst_h, scalix::PixelFormat::Rgba8888, scalix::AllocatorType::Auto));
        }
    } catch (const exception& e) {
        cout << "  [DMA Allocation Not Supported on Host]: " << e.what() << endl;
        return nullopt;
    }

    // 1. Synchronous DMA
    const auto sync_stats = PreciseTimer::measure(num_frames, [&](size_t i) {
        auto src_desc = src_buffers[i]->as_image_desc();
        auto dst_desc = dst_sync_buffers[i]->as_image_desc();
        engine.resize(src_desc, dst_desc, scalix::Filter::Bilinear);
    });

    // 2. Asynchronous DMA
    vector<scalix::Task> async_tasks;
    async_tasks.reserve(num_frames);

    auto async_stats = PreciseTimer::measure(1, [&](size_t) {
        for (size_t i = 0; i < num_frames; ++i) {
            auto src_desc = src_buffers[i]->as_image_desc();
            auto dst_desc = dst_async_buffers[i]->as_image_desc();
            async_tasks.push_back(engine.resize_async(src_desc, dst_desc, scalix::Filter::Bilinear));
        }
        for (size_t i = 0; i < num_frames; ++i) {
            async_tasks[i].wait(0, dst_async_buffers[i]->host_ptr(), dst_async_buffers[i]->size());
        }
    });
    async_stats = MetricStats::from_duration(async_stats.total_ms, num_frames);

    const double speedup = sync_stats.total_ms / async_stats.total_ms;
    cout << "  DMA Synchronous (Blocking):   Total = " << sync_stats.total_ms << " ms, Avg = " << sync_stats.avg_ms << " ms, FPS = " << sync_stats.fps << endl;
    cout << "  DMA Asynchronous (Pipelined): Total = " << async_stats.total_ms << " ms, Avg = " << async_stats.avg_ms << " ms, FPS = " << async_stats.fps << endl;
    cout << "  → Efficiency Gain / Speedup: " << speedup << "x" << endl;

    return ResolutionBenchmarkResult{
        .name = string(name) + " [Zero-Copy DMA]",
        .src_w = src_w,
        .src_h = src_h,
        .dst_w = dst_w,
        .dst_h = dst_h,
        .num_frames = num_frames,
        .sync = sync_stats,
        .async = async_stats,
        .speedup = speedup,
    };
}

// ============================================================================
// Dedicated Sample Image OpenGL Methods Benchmark
// ============================================================================

static StrategyBenchmarkResult benchmark_sample_method_gl(
    scalix::Engine& engine,
    string_view method_name,
    string_view description,
    scalix::Strategy strategy,
    uint32_t max_mip_levels,
    scalix::Filter filter,
    const AlignedVector<uint8_t>& src_buffer,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames,
    string_view output_filename
) {
    const size_t src_stride = static_cast<size_t>(src_w) * 3;
    const size_t dst_stride = static_cast<size_t>(dst_w) * 3;
    const size_t src_size = src_stride * src_h;
    const size_t dst_size = dst_stride * dst_h;

    StrategyBenchmarkResult result{
        .method_name = string(method_name),
        .description = string(description),
        .strategy = strategy,
        .max_mip_levels = max_mip_levels,
        .filter = filter,
        .sync = {},
        .async = {},
        .speedup = 1.0,
        .profile = {},
        .success = false,
        .output_file = string(output_filename),
    };

    const scalix::ResizeOptions options = scalix::ResizeOptions::with_gl(
        filter,
        strategy,
        max_mip_levels
    );

    AlignedVector<uint8_t> dst_sync(dst_size, 0);
    AlignedVector<uint8_t> dst_async(dst_size, 0);

    scalix::ImageDesc src_desc{
        .width = src_w,
        .height = src_h,
        .stride_bytes = src_stride,
        .format = scalix::PixelFormat::Rgb888,
        .host_ptr = const_cast<uint8_t*>(src_buffer.data()),
        .data_len = src_size,
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst_sync_desc{
        .width = dst_w,
        .height = dst_h,
        .stride_bytes = dst_stride,
        .format = scalix::PixelFormat::Rgb888,
        .host_ptr = dst_sync.data(),
        .data_len = dst_size,
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst_async_desc{
        .width = dst_w,
        .height = dst_h,
        .stride_bytes = dst_stride,
        .format = scalix::PixelFormat::Rgb888,
        .host_ptr = dst_async.data(),
        .data_len = dst_size,
        .dma_buf_fd = -1,
    };

    try {
        // Warmup runs
        for (size_t w = 0; w < 2; ++w) {
            engine.resize(src_desc, dst_sync_desc, options);
        }

        // 1. Synchronous Benchmark
        result.sync = PreciseTimer::measure(num_frames, [&](size_t) {
            engine.resize(src_desc, dst_sync_desc, options);
        });

        // 2. Asynchronous Benchmark
        vector<scalix::Task> tasks;
        tasks.reserve(num_frames);

        const auto async_measured = PreciseTimer::measure(1, [&](size_t) {
            for (size_t i = 0; i < num_frames; ++i) {
                tasks.push_back(engine.resize_async(src_desc, dst_async_desc, options));
            }
            for (size_t i = 0; i < num_frames; ++i) {
                tasks[i].wait(0, dst_async.data(), dst_size);
            }
        });
        result.async = MetricStats::from_duration(async_measured.total_ms, num_frames);
        result.speedup = result.sync.total_ms / result.async.total_ms;

        // 3. Stage-by-Stage Profiling
        engine.set_profiling(true);
        double acc_unpack = 0.0, acc_upload = 0.0, acc_scaling = 0.0;
        double acc_download = 0.0, acc_repack = 0.0, acc_sync = 0.0, acc_wall = 0.0;

        for (size_t p = 0; p < PROFILING_ROUNDS; ++p) {
            engine.resize(src_desc, dst_sync_desc, options);
            if (const auto prof = engine.last_profile()) {
                acc_unpack += prof->host_unpack_ms;
                acc_upload += prof->gpu_upload_ms;
                acc_scaling += prof->gpu_pure_blit_ms;
                acc_download += prof->gpu_download_ms;
                acc_repack += prof->host_repack_ms;
                acc_sync += prof->driver_sync_ms;
                acc_wall += prof->total_wall_ms;
            }
        }
        engine.set_profiling(false);

        result.profile.host_unpack_ms = acc_unpack / PROFILING_ROUNDS;
        result.profile.gpu_upload_ms = acc_upload / PROFILING_ROUNDS;
        result.profile.gpu_pure_blit_ms = acc_scaling / PROFILING_ROUNDS;
        result.profile.gpu_download_ms = acc_download / PROFILING_ROUNDS;
        result.profile.host_repack_ms = acc_repack / PROFILING_ROUNDS;
        result.profile.driver_sync_ms = acc_sync / PROFILING_ROUNDS;
        result.profile.total_wall_ms = acc_wall / PROFILING_ROUNDS;

        // Save output verification JPEG
        if (!output_filename.empty()) {
            static_cast<void>(JpegIO::encode_rgb888(output_filename, dst_w, dst_h, dst_sync.data(), dst_stride, VERIFY_JPEG_QUALITY));
        }

        result.success = true;
    } catch (const exception& e) {
        cerr << "  [" << method_name << " Error]: " << e.what() << endl;
        result.success = false;
    }

    return result;
}

// ============================================================================
// Main Execution Entrypoint
// ============================================================================

int main(int argc, char** argv) {
    TableReporter::print_banner(
        "SCALIX OPENGL/GLES HARDWARE ACCELERATOR BENCHMARK SUITE",
        "EGL Headless Multi-Strategy & Resolution Evaluation (SOLID Architecture)"
    );

    // Locate sample image
    string sample_path = string(DEFAULT_SAMPLE_PATH);
    if (argc > 1) {
        sample_path = argv[1];
    } else if (!fs::exists(sample_path) && fs::exists(FALLBACK_SAMPLE_PATH)) {
        sample_path = string(FALLBACK_SAMPLE_PATH);
    }

    const size_t num_sample_rounds = (argc > 2) ? static_cast<size_t>(stoul(argv[2])) : DEFAULT_BENCH_ROUNDS;

    // Initialize Scalix Engine with OpenGL Backend
    unique_ptr<scalix::Engine> engine_ptr;
    try {
        engine_ptr = make_unique<scalix::Engine>(scalix::Backend::OpenGL, PREFIX_BENCH.data());
        cout << "[Engine Initialized: Active Backend = " << engine_ptr->backend_name() << "]" << endl;
    } catch (const exception& e) {
        cerr << "[OpenGL Backend Initialization Failed / Unavailable]: " << e.what() << endl;
        return 1;
    }

    auto& engine = *engine_ptr;

    // 0. Hardware DMA Availability Probe
    bool has_dma = false;
    try {
        scalix::DmaBuffer probe(64, 64, scalix::PixelFormat::Rgba8888, scalix::AllocatorType::Auto);
        has_dma = (probe.fd() >= 0);
    } catch (...) {
        has_dma = false;
    }

    if (has_dma) {
        cout << MSG_DMA_ACTIVE << endl;
    } else {
        cout << MSG_DMA_STAGING << endl;
    }

    // ========================================================================
    // SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK (assets/sample.jpg)
    // ========================================================================
    TableReporter::print_section("SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK: " + sample_path);

    JpegHeader sample_hdr;
    bool sample_loaded = false;
    AlignedVector<uint8_t> sample_rgb_src;
    uint32_t sample_w = 0, sample_h = 0, sample_dst_w = 0, sample_dst_h = 0;

    if (JpegIO::read_header(sample_path, sample_hdr)) {
        sample_w = sample_hdr.width;
        sample_h = sample_hdr.height;
        sample_dst_w = sample_w / DOWNSCALE_RATIO;
        sample_dst_h = sample_h / DOWNSCALE_RATIO;

        const size_t sample_src_stride = sample_hdr.rgb888_stride();
        const size_t sample_src_size = sample_hdr.rgb888_size();
        sample_rgb_src.resize(sample_src_size);

        if (JpegIO::decode_rgb888(sample_path, sample_rgb_src.data(), sample_h, sample_src_stride)) {
            sample_loaded = true;
            cout << "Input Image Probed Successfully:" << endl;
            cout << "  File Path          : " << sample_path << endl;
            cout << "  Dimensions         : " << sample_w << "x" << sample_h << " (Aspect Ratio: "
                 << fixed << setprecision(2) << (static_cast<double>(sample_w) / sample_h) << ")" << endl;
            cout << "  Megapixels         : " << fixed << setprecision(2) << ((sample_w * sample_h) / 1000000.0) << " MP" << endl;
            cout << "  Uncompressed Size  : " << fixed << setprecision(2) << (sample_src_size / (1024.0 * 1024.0)) << " MB (Packed RGB888)" << endl;
            cout << "  Target Resolution  : " << sample_dst_w << "x" << sample_dst_h << " (1/8 Downscaling Thumbnail / NN Input)" << endl;
            cout << "  Benchmark Rounds   : " << num_sample_rounds << " frames per strategy" << endl;
        }
    }

    if (!sample_loaded) {
        cout << "Notice: Sample image " << sample_path << " not found. Generating synthetic 4K pattern." << endl;
        sample_w = SYNTH_4K_W; sample_h = SYNTH_4K_H; sample_dst_w = 480; sample_dst_h = 270;
        sample_rgb_src.resize(static_cast<size_t>(sample_w) * sample_h * 3, 0x77);
        sample_loaded = true;
    }

    if (sample_loaded) {
        cout << "\nExecuting OpenGL Pipeline Strategy Benchmarks on " << sample_path << "..." << endl;

        const vector<StrategyBenchmarkResult> sample_benchmarks = {
            benchmark_sample_method_gl(engine, "1. OpenGL Blit", "Hardware 2D FBO Blitter (`glBlitFramebuffer`)",
                scalix::Strategy::Blit, 0, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_BLIT_FILE),

            benchmark_sample_method_gl(engine, "2. OpenGL Raster", "Offscreen Quad Shader (`glDrawArrays`)",
                scalix::Strategy::Raster, 0, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_RASTER_FILE),

            benchmark_sample_method_gl(engine, "3. OpenGL Lod Mipmap (Auto)", "Hierarchical Mipchain Reduction (Full)",
                scalix::Strategy::LodPyramid, 0, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_LOD_AUTO_FILE),

            benchmark_sample_method_gl(engine, "4. OpenGL Lod Mipmap (2-Lvl)", "Hierarchical Mipchain Reduction (2-Pass)",
                scalix::Strategy::LodPyramid, 2, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_LOD_2LVL_FILE),

            benchmark_sample_method_gl(engine, "5. OpenGL Compute", "Direct Compute Shader Resizer (`glDispatchCompute`)",
                scalix::Strategy::Compute, 0, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_COMPUTE_FILE),

            benchmark_sample_method_gl(engine, "6. OpenGL Raster (Area)", "Offscreen Area Box Averaging Quad Shader (`glDrawArrays`)",
                scalix::Strategy::Raster, 0, scalix::Filter::Area,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_AREA_FILE),

            benchmark_sample_method_gl(engine, "7. OpenGL Auto", "Scalix Engine Adaptive Fast-Path Selector",
                scalix::Strategy::Auto, 0, scalix::Filter::Bilinear,
                sample_rgb_src, sample_w, sample_h, sample_dst_w, sample_dst_h, num_sample_rounds, OUT_AUTO_FILE),
        };

        // Render Dedicated Results Tables
        TableReporter::print_strategy_table(
            "DEDICATED RESULTS: ALL OPENGL METHODS ON " + sample_path + " (" + to_string(sample_w) + "x" + to_string(sample_h) + " → " + to_string(sample_dst_w) + "x" + to_string(sample_dst_h) + ")",
            sample_benchmarks,
            "OpenGL Pipeline Method"
        );

        TableReporter::print_profiling_breakdown_table(
            "STAGE-BY-STAGE GPU LATENCY PROFILING BREAKDOWN (" + sample_path + ")",
            sample_benchmarks,
            "OpenGL Strategy"
        );

        cout << "  Verification artifacts saved to /tmp/bench_gl_sample_*.jpg for visual fidelity inspection." << endl;
    }

#if SCALIX_HAS_OPENCV
    if (sample_loaded) {
        TableReporter::print_section("OPENCV (CPU) vs SCALIX (OPENGL GPU) ON " + sample_path + " (" + to_string(sample_w) + "x" + to_string(sample_h) + " → " + to_string(sample_dst_w) + "x" + to_string(sample_dst_h) + ")");

        struct CvInterpTest final {
            string name;
            int cv_flag;
            scalix::Filter scalix_filter;
        };

        const vector<CvInterpTest> cv_tests = {
            {"Nearest Neighbor", cv::INTER_NEAREST, scalix::Filter::Nearest},
            {"Bilinear", cv::INTER_LINEAR, scalix::Filter::Bilinear},
            {"Bicubic", cv::INTER_CUBIC, scalix::Filter::Bicubic},
            {"Area (Box Average)", cv::INTER_AREA, scalix::Filter::Area},
            {"Lanczos Resampling", cv::INTER_LANCZOS4, scalix::Filter::Lanczos3},
        };

        cv::Mat cv_src_mat(static_cast<int>(sample_h), static_cast<int>(sample_w), CV_8UC3, const_cast<uint8_t*>(sample_rgb_src.data()));
        cv::Mat cv_dst_mat(static_cast<int>(sample_dst_h), static_cast<int>(sample_dst_w), CV_8UC3);
        cv::resize(cv_src_mat, cv_dst_mat, cv::Size(sample_dst_w, sample_dst_h), 0, 0, cv::INTER_LINEAR);

        cout << left << setw(24) << "Interpolation"
             << setw(18) << "OpenCV (CPU) Latency"
             << setw(16) << "OpenCV FPS"
             << setw(18) << "Scalix (GL Async)"
             << setw(16) << "Scalix FPS"
             << "GPU Speedup" << endl;
        cout << "------------------------------------------------------------------------------------------" << endl;

        const size_t src_stride = sample_w * 3;
        const size_t dst_stride = sample_dst_w * 3;
        const size_t src_size = src_stride * sample_h;
        const size_t dst_size = dst_stride * sample_dst_h;
        AlignedVector<uint8_t> sc_dst(dst_size, 0);

        scalix::ImageDesc src_desc{
            .width = sample_w,
            .height = sample_h,
            .stride_bytes = src_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = const_cast<uint8_t*>(sample_rgb_src.data()),
            .data_len = src_size,
            .dma_buf_fd = -1,
        };
        scalix::ImageDesc dst_desc{
            .width = sample_dst_w,
            .height = sample_dst_h,
            .stride_bytes = dst_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = sc_dst.data(),
            .data_len = dst_size,
            .dma_buf_fd = -1,
        };

        for (const auto& test : cv_tests) {
            const auto cv_stats = PreciseTimer::measure(CV_BENCH_ROUNDS, [&](size_t) {
                cv::resize(cv_src_mat, cv_dst_mat, cv::Size(sample_dst_w, sample_dst_h), 0, 0, test.cv_flag);
            });

            vector<scalix::Task> sc_tasks;
            sc_tasks.reserve(CV_BENCH_ROUNDS);
            auto sc_stats = PreciseTimer::measure(1, [&](size_t) {
                for (size_t r = 0; r < CV_BENCH_ROUNDS; ++r) {
                    sc_tasks.push_back(engine.resize_async(src_desc, dst_desc, test.scalix_filter));
                }
                for (size_t r = 0; r < CV_BENCH_ROUNDS; ++r) {
                    sc_tasks[r].wait(0, sc_dst.data(), sc_dst.size());
                }
            });
            sc_stats = MetricStats::from_duration(sc_stats.total_ms, CV_BENCH_ROUNDS);

            const double speedup = cv_stats.avg_ms / sc_stats.avg_ms;

            cout << left << setw(24) << test.name
                 << setw(18) << (to_string(cv_stats.avg_ms).substr(0, 6) + UNIT_MS.data())
                 << setw(16) << (to_string(cv_stats.fps).substr(0, 6) + UNIT_FPS.data())
                 << setw(18) << (to_string(sc_stats.avg_ms).substr(0, 6) + UNIT_MS.data())
                 << setw(16) << (to_string(sc_stats.fps).substr(0, 6) + UNIT_FPS.data())
                 << (to_string(speedup).substr(0, 5) + UNIT_X.data()) << endl;
        }
        cout << "==========================================================================================" << endl;
    }
#endif

    // ========================================================================
    // SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888)
    // ========================================================================
    TableReporter::print_section("SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888 → 320x320)");

    vector<ResolutionBenchmarkResult> synth_results = {
        run_resolution_benchmark(engine, "4K UHD (3840x2160)", SYNTH_4K_W, SYNTH_4K_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS),
        run_resolution_benchmark(engine, "Full HD (1920x1080)", SYNTH_FHD_W, SYNTH_FHD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS),
        run_resolution_benchmark(engine, "HD 720p (1280x720)", SYNTH_HD_W, SYNTH_HD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS),
    };

    // Section 3: Zero-Copy Hardware DMA Buffer Benchmark
    if (has_dma) {
        TableReporter::print_section("SECTION 3: ZERO-COPY HARDWARE DMA BENCHMARKS (RGBA8888)");
        if (auto m4k = run_dma_resolution_benchmark(engine, "4K UHD (3840x2160)", SYNTH_4K_W, SYNTH_4K_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS)) synth_results.push_back(*m4k);
        if (auto mfhd = run_dma_resolution_benchmark(engine, "Full HD (1920x1080)", SYNTH_FHD_W, SYNTH_FHD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS)) synth_results.push_back(*mfhd);
        if (auto mhd = run_dma_resolution_benchmark(engine, "HD 720p (1280x720)", SYNTH_HD_W, SYNTH_HD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, DEFAULT_BENCH_ROUNDS)) synth_results.push_back(*mhd);
    }

    TableReporter::print_resolution_summary_table("SYNTHETIC MULTI-RESOLUTION BENCHMARK COMPARISON SUMMARY (OPENGL)", synth_results);

    cout << "\n[OpenGL benchmark suite completed successfully!]" << endl;
    return 0;
}
