#include "benchmark_utils.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <cassert>
#include <cstring>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

namespace scalix::examples {

namespace {

constexpr int WIDTH_METHOD_NAME = 36;
constexpr int WIDTH_RES_NAME    = 24;
constexpr int WIDTH_STRAT_NAME  = 36;
constexpr int WIDTH_TARGET      = 18;
constexpr int WIDTH_LATENCY     = 15;
constexpr int WIDTH_FPS         = 13;
constexpr int WIDTH_SHORT_COL   = 12;
constexpr int WIDTH_MED_COL     = 14;

constexpr string_view UNIT_MS   = " ms";
constexpr string_view UNIT_FPS  = " FPS";
constexpr string_view UNIT_X    = "x";

} // anonymous namespace

MetricStats MetricStats::from_duration(double total_duration_ms, size_t count) {
    const double avg = (count > 0) ? (total_duration_ms / count) : 0.0;
    const double fps_val = (total_duration_ms > 0.0) ? ((count / total_duration_ms) * 1000.0) : 0.0;
    return MetricStats{
        .total_ms = total_duration_ms,
        .avg_ms = avg,
        .fps = fps_val,
    };
}

void TableReporter::print_banner(string_view title, string_view subtitle) {
    cout << BANNER_DIVIDER << endl;
    cout << "  " << title << endl;
    if (!subtitle.empty()) {
        cout << "  " << subtitle << endl;
    }
    cout << BANNER_DIVIDER << endl;
}

void TableReporter::print_section(string_view section_title) {
    cout << "\n" << BANNER_DIVIDER << endl;
    cout << section_title << endl;
    cout << BANNER_DIVIDER << endl;
}

void TableReporter::print_strategy_table(
    string_view title,
    const vector<StrategyBenchmarkResult>& results,
    string_view method_header
) {
    cout << "\n" << BANNER_DIVIDER << endl;
    cout << "     " << title << endl;
    cout << BANNER_DIVIDER << endl;
    cout << left << setw(WIDTH_METHOD_NAME) << method_header
         << setw(WIDTH_LATENCY) << "Sync Latency"
         << setw(WIDTH_FPS) << "Sync FPS"
         << setw(WIDTH_LATENCY) << "Async Latency"
         << setw(WIDTH_FPS) << "Async FPS"
         << "Async Gain" << endl;
    cout << SECTION_DIVIDER << endl;

    for (const auto& r : results) {
        if (!r.success) continue;
        cout << left << setw(WIDTH_METHOD_NAME) << r.method_name
             << setw(WIDTH_LATENCY) << (to_string(r.sync.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(WIDTH_FPS) << (to_string(r.sync.fps).substr(0, 6) + UNIT_FPS.data())
             << setw(WIDTH_LATENCY) << (to_string(r.async.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(WIDTH_FPS) << (to_string(r.async.fps).substr(0, 6) + UNIT_FPS.data())
             << (to_string(r.speedup).substr(0, 5) + UNIT_X.data()) << endl;
    }
    cout << BANNER_DIVIDER << endl;
}

void TableReporter::print_profiling_breakdown_table(
    string_view title,
    const vector<StrategyBenchmarkResult>& results,
    string_view strategy_header
) {
    cout << "\n" << BANNER_DIVIDER << endl;
    cout << "          " << title << endl;
    cout << BANNER_DIVIDER << endl;
    cout << left << setw(WIDTH_STRAT_NAME) << strategy_header
         << setw(WIDTH_SHORT_COL) << "Upload"
         << setw(WIDTH_MED_COL) << "GPU Core"
         << setw(WIDTH_SHORT_COL) << "Download"
         << setw(WIDTH_SHORT_COL) << "Sync Wait"
         << setw(WIDTH_MED_COL) << "Total Wall"
         << "Pure GPU FPS" << endl;
    cout << SECTION_DIVIDER << endl;

    for (const auto& r : results) {
        if (!r.success) continue;
        const double pure_gpu_ms = r.profile.gpu_pure_blit_ms;
        const double pure_gpu_fps = (pure_gpu_ms > 0.0) ? (1000.0 / pure_gpu_ms) : 0.0;

        cout << left << setw(WIDTH_STRAT_NAME) << r.method_name
             << setw(WIDTH_SHORT_COL) << (to_string(r.profile.gpu_upload_ms).substr(0, 5) + UNIT_MS.data())
             << setw(WIDTH_MED_COL) << (to_string(r.profile.gpu_pure_blit_ms).substr(0, 5) + UNIT_MS.data())
             << setw(WIDTH_SHORT_COL) << (to_string(r.profile.gpu_download_ms).substr(0, 5) + UNIT_MS.data())
             << setw(WIDTH_SHORT_COL) << (to_string(r.profile.driver_sync_ms).substr(0, 5) + UNIT_MS.data())
             << setw(WIDTH_MED_COL) << (to_string(r.profile.total_wall_ms).substr(0, 5) + UNIT_MS.data())
             << (to_string(pure_gpu_fps).substr(0, 6) + UNIT_FPS.data()) << endl;
    }
    cout << BANNER_DIVIDER << endl;
}

void TableReporter::print_resolution_summary_table(
    string_view title,
    const vector<ResolutionBenchmarkResult>& results
) {
    cout << "\n" << BANNER_DIVIDER << endl;
    cout << "              " << title << endl;
    cout << BANNER_DIVIDER << endl;
    cout << left << setw(WIDTH_RES_NAME) << "Resolution"
         << setw(WIDTH_TARGET) << "Target"
         << setw(WIDTH_LATENCY) << "Sync Latency"
         << setw(WIDTH_FPS) << "Sync FPS"
         << setw(WIDTH_LATENCY) << "Async Latency"
         << setw(WIDTH_FPS) << "Async FPS"
         << "Speedup" << endl;
    cout << SECTION_DIVIDER << endl;

    for (const auto& r : results) {
        cout << left << setw(WIDTH_RES_NAME) << r.name
             << setw(WIDTH_TARGET) << ("→ " + to_string(r.dst_w) + "x" + to_string(r.dst_h))
             << setw(WIDTH_LATENCY) << (to_string(r.sync.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(WIDTH_FPS) << (to_string(r.sync.fps).substr(0, 6) + UNIT_FPS.data())
             << setw(WIDTH_LATENCY) << (to_string(r.async.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(WIDTH_FPS) << (to_string(r.async.fps).substr(0, 6) + UNIT_FPS.data())
             << (to_string(r.speedup).substr(0, 5) + UNIT_X.data()) << endl;
    }
    cout << BANNER_DIVIDER << endl;
}

void probe_and_print_dma_status() {
    bool has_dma = false;
    try {
        scalix::DmaBuffer probe(64, 64, scalix::PixelFormat::Rgba8888);
        has_dma = (probe.fd() >= 0);
    } catch (...) {
        has_dma = false;
    }

    if (has_dma) {
        cout << MSG_DMA_ACTIVE << endl;
    } else {
        cout << MSG_DMA_STAGING << endl;
    }
}

SampleImageData load_or_generate_sample(
    string_view requested_path,
    size_t num_rounds,
    uint32_t downscale_ratio
) {
    SampleImageData result{};
    string sample_path = string(requested_path);
    if (!fs::exists(sample_path) && fs::exists(FALLBACK_SAMPLE_PATH)) {
        sample_path = string(FALLBACK_SAMPLE_PATH);
    }
    result.path = sample_path;

    JpegHeader sample_hdr;
    bool sample_loaded = false;

    if (JpegIO::read_header(sample_path, sample_hdr)) {
        result.width = sample_hdr.width;
        result.height = sample_hdr.height;
        result.dst_width = result.width / downscale_ratio;
        result.dst_height = result.height / downscale_ratio;

        const size_t sample_src_stride = sample_hdr.rgb888_stride();
        const size_t sample_src_size = sample_hdr.rgb888_size();
        result.rgb_data.resize(sample_src_size);

        if (JpegIO::decode_rgb888(sample_path, result.rgb_data.data(), result.height, sample_src_stride)) {
            sample_loaded = true;
            cout << "Input Image Probed Successfully:" << endl;
            cout << "  File Path          : " << sample_path << endl;
            cout << "  Dimensions         : " << result.width << "x" << result.height << " (Aspect Ratio: "
                 << fixed << setprecision(2) << (static_cast<double>(result.width) / result.height) << ")" << endl;
            cout << "  Megapixels         : " << fixed << setprecision(2) << ((result.width * result.height) / 1000000.0) << " MP" << endl;
            cout << "  Uncompressed Size  : " << fixed << setprecision(2) << (sample_src_size / (1024.0 * 1024.0)) << " MB (Packed RGB888)" << endl;
            cout << "  Target Resolution  : " << result.dst_width << "x" << result.dst_height << " (1/" << downscale_ratio << " Downscaling Thumbnail / NN Input)" << endl;
            cout << "  Benchmark Rounds   : " << num_rounds << " frames per strategy" << endl;
        }
    }

    if (!sample_loaded) {
        cout << "Notice: Sample image " << sample_path << " not found. Generating synthetic 4K pattern." << endl;
        result.width = SYNTH_4K_W;
        result.height = SYNTH_4K_H;
        result.dst_width = 480;
        result.dst_height = 270;
        result.rgb_data.resize(static_cast<size_t>(result.width) * result.height * 3, 0x77);
        result.is_synthetic = true;
    }

    return result;
}

StrategyBenchmarkResult benchmark_strategy(
    scalix::Engine& engine,
    string_view method_name,
    string_view description,
    const scalix::ResizeOptions& options,
    const scalix::AlignedVector<uint8_t>& src_buffer,
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
        .strategy = scalix::Strategy::Auto,
        .max_mip_levels = 0,
        .filter = scalix::Filter::Bilinear,
        .sync = {},
        .async = {},
        .speedup = 1.0,
        .profile = {},
        .success = false,
        .output_file = string(output_filename),
    };

    scalix::AlignedVector<uint8_t> dst_sync(dst_size, 0);
    scalix::AlignedVector<uint8_t> dst_async(dst_size, 0);

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
        result.speedup = (result.async.total_ms > 0.0) ? (result.sync.total_ms / result.async.total_ms) : 1.0;

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

static ResolutionBenchmarkResult run_single_resolution_benchmark(
    scalix::Engine& engine,
    string_view name,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames
) {
    const size_t src_stride = static_cast<size_t>(src_w) * 3;
    const size_t dst_stride = static_cast<size_t>(dst_w) * 3;
    const size_t src_size = src_stride * src_h;
    const size_t dst_size = dst_stride * dst_h;

    cout << "\n--------------------------------------------------------" << endl;
    cout << "[Benchmark: " << name << " (" << src_w << "x" << src_h << " → " << dst_w << "x" << dst_h << ", RGB888, " << num_frames << " frames)]" << endl;
    cout << "  Memory per 1 frame: " << (src_size / (1024.0 * 1024.0)) << " MB (Total: "
         << ((src_size * num_frames) / (1024.0 * 1024.0)) << " MB)" << endl;

    vector<scalix::AlignedVector<uint8_t>> src_buffers(num_frames, scalix::AlignedVector<uint8_t>(src_size));
    vector<scalix::AlignedVector<uint8_t>> dst_sync_buffers(num_frames, scalix::AlignedVector<uint8_t>(dst_size, 0));
    vector<scalix::AlignedVector<uint8_t>> dst_async_buffers(num_frames, scalix::AlignedVector<uint8_t>(dst_size, 0));

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
    const double speedup = (async_stats.total_ms > 0.0) ? (sync_stats.total_ms / async_stats.total_ms) : 1.0;

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

static optional<ResolutionBenchmarkResult> run_single_dma_benchmark(
    scalix::Engine& engine,
    string_view name,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames
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
            auto src_buf = make_unique<scalix::DmaBuffer>(src_w, src_h, scalix::PixelFormat::Rgba8888);
            const uint8_t pattern = static_cast<uint8_t>((i * 17 + 0x33) & 0xFF);
            src_buf->with_write([pattern](uint8_t* ptr, size_t size) {
                if (ptr && size > 0) memset(ptr, pattern, size);
            });
            src_buffers.push_back(move(src_buf));
            dst_sync_buffers.push_back(make_unique<scalix::DmaBuffer>(dst_w, dst_h, scalix::PixelFormat::Rgba8888));
            dst_async_buffers.push_back(make_unique<scalix::DmaBuffer>(dst_w, dst_h, scalix::PixelFormat::Rgba8888));
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

    const double speedup = (async_stats.total_ms > 0.0) ? (sync_stats.total_ms / async_stats.total_ms) : 1.0;
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

void run_synthetic_benchmarks_suite(
    scalix::Engine& engine,
    size_t num_frames,
    string_view summary_title
) {
    vector<ResolutionBenchmarkResult> results;
    results.push_back(run_single_resolution_benchmark(engine, "4K UHD (3840x2160)", SYNTH_4K_W, SYNTH_4K_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames));
    results.push_back(run_single_resolution_benchmark(engine, "Full HD (1920x1080)", SYNTH_FHD_W, SYNTH_FHD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames));
    results.push_back(run_single_resolution_benchmark(engine, "HD 720p (1280x720)", SYNTH_HD_W, SYNTH_HD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames));

    if (auto dma_4k = run_single_dma_benchmark(engine, "4K UHD (3840x2160)", SYNTH_4K_W, SYNTH_4K_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames)) {
        results.push_back(move(*dma_4k));
    }
    if (auto dma_fhd = run_single_dma_benchmark(engine, "Full HD (1920x1080)", SYNTH_FHD_W, SYNTH_FHD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames)) {
        results.push_back(move(*dma_fhd));
    }
    if (auto dma_hd = run_single_dma_benchmark(engine, "HD 720p (1280x720)", SYNTH_HD_W, SYNTH_HD_H, SYNTH_DST_DIM, SYNTH_DST_DIM, num_frames)) {
        results.push_back(move(*dma_hd));
    }

    TableReporter::print_resolution_summary_table(summary_title, results);
}

#if SCALIX_HAS_OPENCV
void run_opencv_comparison_suite(
    scalix::Engine& engine,
    const SampleImageData& sample,
    string_view backend_display_name,
    const function<scalix::ResizeOptions(scalix::Filter)>& options_factory,
    size_t num_rounds
) {
    TableReporter::print_section("OPENCV (CPU) vs SCALIX (" + string(backend_display_name) + " GPU) ON " + sample.path + " (" + to_string(sample.width) + "x" + to_string(sample.height) + " → " + to_string(sample.dst_width) + "x" + to_string(sample.dst_height) + ")");

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

    cv::Mat cv_src_mat(static_cast<int>(sample.height), static_cast<int>(sample.width), CV_8UC3, const_cast<uint8_t*>(sample.rgb_data.data()));
    cv::Mat cv_dst_mat(static_cast<int>(sample.dst_height), static_cast<int>(sample.dst_width), CV_8UC3);

    // Warmup OpenCV
    cv::resize(cv_src_mat, cv_dst_mat, cv::Size(sample.dst_width, sample.dst_height), 0, 0, cv::INTER_LINEAR);

    cout << left << setw(24) << "Interpolation"
         << setw(20) << "OpenCV (CPU) Latency"
         << setw(16) << "OpenCV FPS"
         << setw(24) << ("Scalix (" + string(backend_display_name) + " Async)")
         << setw(16) << "Scalix FPS"
         << "GPU Speedup" << endl;
    cout << SECTION_DIVIDER << endl;

    for (const auto& test : cv_tests) {
        // Measure OpenCV
        const auto cv_stats = PreciseTimer::measure(num_rounds, [&](size_t) {
            cv::resize(cv_src_mat, cv_dst_mat, cv::Size(sample.dst_width, sample.dst_height), 0, 0, test.cv_flag);
        });

        // Measure Scalix
        const auto scalix_opts = options_factory(test.scalix_filter);
        const auto scalix_res = benchmark_strategy(
            engine, test.name, "", scalix_opts,
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height,
            num_rounds, ""
        );

        const double cv_gpu_speedup = (scalix_res.async.avg_ms > 0.0) ? (cv_stats.avg_ms / scalix_res.async.avg_ms) : 0.0;

        cout << left << setw(24) << test.name
             << setw(20) << (to_string(cv_stats.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(16) << (to_string(cv_stats.fps).substr(0, 6) + UNIT_FPS.data())
             << setw(24) << (to_string(scalix_res.async.avg_ms).substr(0, 6) + UNIT_MS.data())
             << setw(16) << (to_string(scalix_res.async.fps).substr(0, 6) + UNIT_FPS.data())
             << (to_string(cv_gpu_speedup).substr(0, 5) + UNIT_X.data()) << endl;
    }
    cout << BANNER_DIVIDER << endl;
}
#endif

} // namespace scalix::examples
