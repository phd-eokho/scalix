#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <scalix/scalix.hpp>
#include "benchmark_utils.hpp"

using namespace std;
using namespace scalix::examples;

namespace {

constexpr string_view OUT_NEAREST_BLIT_FILE     = "/tmp/bench_vulkan_sample_nearest_blit.jpg";
constexpr string_view OUT_NEAREST_RASTER_FILE   = "/tmp/bench_vulkan_sample_nearest_raster.jpg";
constexpr string_view OUT_NEAREST_COMPUTE_FILE  = "/tmp/bench_vulkan_sample_nearest_compute.jpg";

constexpr string_view OUT_BILINEAR_BLIT_FILE     = "/tmp/bench_vulkan_sample_bilinear_blit.jpg";
constexpr string_view OUT_BILINEAR_RASTER_FILE   = "/tmp/bench_vulkan_sample_bilinear_raster.jpg";
constexpr string_view OUT_BILINEAR_COMPUTE_FILE  = "/tmp/bench_vulkan_sample_bilinear_compute.jpg";

constexpr string_view OUT_LOD_FULL_FILE          = "/tmp/bench_vulkan_sample_lod_full.jpg";
constexpr string_view OUT_LOD_2LVL_FILE          = "/tmp/bench_vulkan_sample_lod_2lvl.jpg";

constexpr string_view OUT_BICUBIC_COMPUTE_FILE  = "/tmp/bench_vulkan_sample_bicubic_compute.jpg";
constexpr string_view OUT_LANCZOS3_COMPUTE_FILE = "/tmp/bench_vulkan_sample_lanczos3_compute.jpg";

constexpr string_view OUT_AREA_COMPUTE_FILE     = "/tmp/bench_vulkan_sample_area_compute.jpg";

constexpr string_view OUT_AUTO_FILE             = "/tmp/bench_vulkan_sample_auto.jpg";

} // anonymous namespace

int main(int argc, char** argv) {
    TableReporter::print_banner(
        "SCALIX VULKAN HARDWARE ACCELERATOR BENCHMARK SUITE",
        "Categorized Filter Modes & Multi-Strategy Hardware Evaluation (SOLID Architecture)"
    );

    const string sample_path = (argc > 1) ? argv[1] : string(DEFAULT_SAMPLE_PATH);
    const size_t num_sample_rounds = (argc > 2) ? static_cast<size_t>(stoul(argv[2])) : DEFAULT_BENCH_ROUNDS;

    // Initialize Scalix Engine with Vulkan Hardware Backend
    scalix::Engine engine(scalix::Backend::Vulkan, "bench_vulkan");

    // 0. Hardware DMA Availability Probe
    probe_and_print_dma_status();

    // ========================================================================
    // SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK (assets/sample.jpg)
    // ========================================================================
    TableReporter::print_section("SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK: " + sample_path);
    const auto sample = load_or_generate_sample(sample_path, num_sample_rounds);

    cout << "\nExecuting Vulkan Pipeline Strategy Benchmarks on " << sample.path << "..." << endl;

    const vector<StrategyBenchmarkResult> sample_benchmarks = {
        // --- 1. NEAREST NEIGHBOR (0-ORDER HOLD) ---
        benchmark_strategy(engine, "1. Nearest (Vulkan Blit)", "Hardware 2D Fixed-Function Blitter (`vkCmdBlitImage`, Nearest)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Nearest, scalix::Strategy::Blit),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_BLIT_FILE),

        benchmark_strategy(engine, "2. Nearest (Vulkan Raster)", "Offscreen Raster Graphics Quad (`vkCmdDraw`, Nearest)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Nearest, scalix::Strategy::Raster),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_RASTER_FILE),

        benchmark_strategy(engine, "3. Nearest (Vulkan Compute)", "Direct Fused 24-bit Compute Resizer (`vkCmdDispatch`, Nearest)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Nearest, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_COMPUTE_FILE),

        // --- 2. BILINEAR INTERPOLATION (1ST-ORDER TENT FILTER) ---
        benchmark_strategy(engine, "4. Bilinear (Vulkan Blit)", "Hardware 2D Fixed-Function Blitter (`vkCmdBlitImage`, Bilinear)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::Blit),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_BLIT_FILE),

        benchmark_strategy(engine, "5. Bilinear (Vulkan Raster)", "Offscreen Raster Graphics Quad (`vkCmdDraw`, Bilinear)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::Raster),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_RASTER_FILE),

        benchmark_strategy(engine, "6. Bilinear (Vulkan Compute)", "Direct Fused 24-bit Compute Resizer (`vkCmdDispatch`, Bilinear)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_COMPUTE_FILE),

        // --- 3. HIERARCHICAL MIPMAP / LOD PYRAMID DOWNSCALING ---
        benchmark_strategy(engine, "7. Lod Mipmap (Vulkan Full)", "Hierarchical Multi-Pass Mipchain Downscaler (Full Depth)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::LodPyramid, 0),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_LOD_FULL_FILE),

        benchmark_strategy(engine, "8. Lod Mipmap (Vulkan 2-Pass)", "Hierarchical Multi-Pass Mipchain Downscaler (2-Pass Limit)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::LodPyramid, 2),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_LOD_2LVL_FILE),

        // --- 4. BICUBIC SPLINE (CATMULL-ROM 4x4) ---
        benchmark_strategy(engine, "9. Bicubic (Vulkan Compute)", "Direct Fused 24-bit 4x4 Spline Resizer (`vkCmdDispatch`, Bicubic)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bicubic, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BICUBIC_COMPUTE_FILE),

        // --- 5. LANCZOS-3 (3-LOBE SINC 6x6) ---
        benchmark_strategy(engine, "10. Lanczos-3 (Vulkan Compute)", "Direct Fused 24-bit 6x6 Windowed Sinc (`vkCmdDispatch`, Lanczos3)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Lanczos3, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_LANCZOS3_COMPUTE_FILE),

        // --- 6. AREA (BOX AVERAGE / PIXEL OVERLAP) ---
        benchmark_strategy(engine, "11. Area (Vulkan Compute)", "Direct Fused 24-bit Area Box Averaging Resizer (`vkCmdDispatch`, Area)",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Area, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AREA_COMPUTE_FILE),

        // --- 7. ADAPTIVE ENGINE SELECTOR ---
        benchmark_strategy(engine, "12. Auto (Vulkan Adaptive)", "Scalix Engine Adaptive Fast-Path Strategy Selector",
            scalix::ResizeOptions::with_vulkan(scalix::Filter::Bilinear, scalix::Strategy::Auto),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AUTO_FILE),
    };

    // Render Dedicated Results Tables
    TableReporter::print_vulkan_strategy_table(
        "DEDICATED RESULTS: ALL VULKAN METHODS ON " + sample.path + " (" + to_string(sample.width) + "x" + to_string(sample.height) + " → " + to_string(sample.dst_width) + "x" + to_string(sample.dst_height) + ")",
        sample_benchmarks
    );

    TableReporter::print_profiling_breakdown_table(
        "STAGE-BY-STAGE GPU LATENCY PROFILING BREAKDOWN (" + sample.path + ")",
        sample_benchmarks,
        "Vulkan Hardware Strategy"
    );

    cout << "  Verification artifacts saved to /tmp/bench_vulkan_sample_*.jpg for visual fidelity inspection." << endl;

#if SCALIX_HAS_OPENCV
    run_opencv_comparison_suite(engine, sample, "Vulkan", [](scalix::Filter filter) {
        return scalix::ResizeOptions::with_vulkan(filter, scalix::Strategy::Compute);
    });
#endif

    // ========================================================================
    // SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888 → 320x320)
    // ========================================================================
    TableReporter::print_section("SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888 → 320x320)");
    run_synthetic_benchmarks_suite(engine, DEFAULT_BENCH_ROUNDS, "SYNTHETIC MULTI-RESOLUTION BENCHMARK COMPARISON SUMMARY (VULKAN)");

    cout << "\n[Vulkan benchmark suite completed successfully!]" << endl;
    return 0;
}
