#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <scalix/scalix.hpp>
#include "benchmark_utils.hpp"

using namespace std;
using namespace scalix::examples;

namespace {

constexpr string_view OUT_NEAREST_FILE  = "/tmp/bench_opencl_sample_nearest.jpg";
constexpr string_view OUT_BILINEAR_FILE = "/tmp/bench_opencl_sample_bilinear.jpg";
constexpr string_view OUT_BICUBIC_FILE  = "/tmp/bench_opencl_sample_bicubic.jpg";
constexpr string_view OUT_LANCZOS3_FILE = "/tmp/bench_opencl_sample_lanczos3.jpg";
constexpr string_view OUT_AREA_FILE     = "/tmp/bench_opencl_sample_area.jpg";
constexpr string_view OUT_AUTO_FILE     = "/tmp/bench_opencl_sample_auto.jpg";

} // anonymous namespace

int main(int argc, char** argv) {
    TableReporter::print_banner(
        "SCALIX OPENCL HARDWARE ACCELERATOR BENCHMARK SUITE",
        "Categorized Filter Modes & Direct Compute Hardware Evaluation (SOLID Architecture)"
    );

    const string sample_path = (argc > 1) ? argv[1] : string(DEFAULT_SAMPLE_PATH);
    const size_t num_sample_rounds = (argc > 2) ? static_cast<size_t>(stoul(argv[2])) : DEFAULT_BENCH_ROUNDS;

    // Initialize Scalix Engine with OpenCL Hardware Backend
    scalix::Engine engine(scalix::Backend::OpenCL, "bench_opencl");

    // 0. Hardware DMA Availability Probe
    probe_and_print_dma_status();

    // ========================================================================
    // SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK (assets/sample.jpg)
    // ========================================================================
    TableReporter::print_section("SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK: " + sample_path);
    const auto sample = load_or_generate_sample(sample_path, num_sample_rounds);

    cout << "\nExecuting OpenCL Compute Filter Benchmarks on " << sample.path << "..." << endl;

    const vector<StrategyBenchmarkResult> sample_benchmarks = {
        // --- 1. NEAREST NEIGHBOR (0-ORDER HOLD) ---
        benchmark_strategy(engine, "1. Nearest (OpenCL Compute)", "Direct Flat Buffer Compute (0-Order Hold, `clEnqueueNDRangeKernel`)",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Nearest, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_FILE),

        // --- 2. BILINEAR INTERPOLATION (1ST-ORDER TENT FILTER) ---
        benchmark_strategy(engine, "2. Bilinear (OpenCL Compute)", "Direct Flat Buffer Compute (1st-Order Resampler, `clEnqueueNDRangeKernel`)",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Bilinear, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_FILE),

        // --- 3. BICUBIC SPLINE (CATMULL-ROM 4x4) ---
        benchmark_strategy(engine, "3. Bicubic (OpenCL Compute)", "Direct Flat Buffer Compute (Catmull-Rom 4x4 Spline, `clEnqueueNDRangeKernel`)",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Bicubic, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BICUBIC_FILE),

        // --- 4. LANCZOS-3 (3-LOBE SINC 6x6) ---
        benchmark_strategy(engine, "4. Lanczos-3 (OpenCL Compute)", "Direct Flat Buffer Compute (3-Lobe Sinc Window 6x6, `clEnqueueNDRangeKernel`)",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Lanczos3, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_LANCZOS3_FILE),

        // --- 5. AREA (BOX AVERAGE / OVERLAP) ---
        benchmark_strategy(engine, "5. Area (OpenCL Compute)", "Direct Flat Buffer Compute (Subpixel 2D Box Overlap, `clEnqueueNDRangeKernel`)",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Area, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AREA_FILE),

        // --- 6. ADAPTIVE ENGINE SELECTOR ---
        benchmark_strategy(engine, "6. Auto (OpenCL Adaptive)", "Scalix Engine Adaptive Fast-Path Selector",
            scalix::ResizeOptions::with_opencl(scalix::Filter::Bilinear, scalix::Strategy::Auto),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AUTO_FILE),
    };

    // Render Dedicated Results Tables
    TableReporter::print_opencl_strategy_table(
        "DEDICATED RESULTS: ALL OPENCL METHODS ON " + sample.path + " (" + to_string(sample.width) + "x" + to_string(sample.height) + " → " + to_string(sample.dst_width) + "x" + to_string(sample.dst_height) + ")",
        sample_benchmarks
    );

    TableReporter::print_profiling_breakdown_table(
        "STAGE-BY-STAGE GPU LATENCY PROFILING BREAKDOWN (" + sample.path + ")",
        sample_benchmarks,
        "OpenCL Hardware Strategy"
    );

    cout << "  Verification artifacts saved to /tmp/bench_opencl_sample_*.jpg for visual fidelity inspection." << endl;

#if SCALIX_HAS_OPENCV
    run_opencv_comparison_suite(engine, sample, "OpenCL", [](scalix::Filter filter) {
        return scalix::ResizeOptions::with_opencl(filter, scalix::Strategy::Compute);
    });
#endif

    // ========================================================================
    // SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888 → 320x320)
    // ========================================================================
    TableReporter::print_section("SECTION 2: MULTI-RESOLUTION DOWNSCALING TO TENSOR (320x320, RGB888)");
    run_synthetic_benchmarks_suite(engine, DEFAULT_BENCH_ROUNDS, "MULTI-RESOLUTION RESOLUTION SCALING THROUGHPUT SUMMARY (OPENCL)");

    cout << "\n[OpenCL benchmark suite completed successfully!]" << endl;
    return 0;
}
