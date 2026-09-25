#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <scalix/scalix.hpp>
#include "benchmark_utils.hpp"

using namespace std;
using namespace scalix::examples;

namespace {

constexpr string_view OUT_NEAREST_BLIT_FILE     = "/tmp/bench_gl_sample_nearest_blit.jpg";
constexpr string_view OUT_NEAREST_RASTER_FILE   = "/tmp/bench_gl_sample_nearest_raster.jpg";
constexpr string_view OUT_NEAREST_COMPUTE_FILE  = "/tmp/bench_gl_sample_nearest_compute.jpg";

constexpr string_view OUT_BILINEAR_BLIT_FILE     = "/tmp/bench_gl_sample_bilinear_blit.jpg";
constexpr string_view OUT_BILINEAR_RASTER_FILE   = "/tmp/bench_gl_sample_bilinear_raster.jpg";
constexpr string_view OUT_BILINEAR_LOD_AUTO_FILE = "/tmp/bench_gl_sample_bilinear_lod_auto.jpg";
constexpr string_view OUT_BILINEAR_LOD_2LVL_FILE = "/tmp/bench_gl_sample_bilinear_lod_2lvl.jpg";
constexpr string_view OUT_BILINEAR_COMPUTE_FILE  = "/tmp/bench_gl_sample_bilinear_compute.jpg";

constexpr string_view OUT_BICUBIC_COMPUTE_FILE  = "/tmp/bench_gl_sample_bicubic_compute.jpg";
constexpr string_view OUT_LANCZOS3_COMPUTE_FILE = "/tmp/bench_gl_sample_lanczos3_compute.jpg";

constexpr string_view OUT_AREA_RASTER_FILE      = "/tmp/bench_gl_sample_area_raster.jpg";
constexpr string_view OUT_AREA_COMPUTE_FILE     = "/tmp/bench_gl_sample_area_compute.jpg";

constexpr string_view OUT_AUTO_FILE             = "/tmp/bench_gl_sample_auto.jpg";

} // anonymous namespace

int main(int argc, char** argv) {
    TableReporter::print_banner(
        "SCALIX OPENGL / EGL HARDWARE ACCELERATOR BENCHMARK SUITE",
        "Categorized Filter Modes & Multi-Strategy Hardware Evaluation (SOLID Architecture)"
    );

    const string sample_path = (argc > 1) ? argv[1] : string(DEFAULT_SAMPLE_PATH);
    const size_t num_sample_rounds = (argc > 2) ? static_cast<size_t>(stoul(argv[2])) : DEFAULT_BENCH_ROUNDS;

    // Initialize Scalix Engine with OpenGL Hardware Backend
    scalix::Engine engine(scalix::Backend::OpenGL, "bench_gl");

    // 0. Hardware DMA Availability Probe
    probe_and_print_dma_status();

    // ========================================================================
    // SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK (assets/sample.jpg)
    // ========================================================================
    TableReporter::print_section("SECTION 1: DEDICATED REAL-WORLD IMAGE BENCHMARK: " + sample_path);
    const auto sample = load_or_generate_sample(sample_path, num_sample_rounds);

    cout << "\nExecuting OpenGL Pipeline Strategy Benchmarks on " << sample.path << "..." << endl;

    const vector<StrategyBenchmarkResult> sample_benchmarks = {
        // --- 1. NEAREST NEIGHBOR (0-ORDER HOLD) ---
        benchmark_strategy(engine, "1. Nearest (OpenGL Blit)", "Hardware 2D FBO Blitter (`glBlitFramebuffer`, Nearest)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Nearest, scalix::Strategy::Blit),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_BLIT_FILE),

        benchmark_strategy(engine, "2. Nearest (OpenGL Raster)", "Offscreen Quad Shader (`glDrawArrays`, Nearest)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Nearest, scalix::Strategy::Raster),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_RASTER_FILE),

        benchmark_strategy(engine, "3. Nearest (OpenGL Compute)", "Direct Compute Shader Resizer (`glDispatchCompute`, Nearest)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Nearest, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_NEAREST_COMPUTE_FILE),

        // --- 2. BILINEAR INTERPOLATION (1ST-ORDER TENT FILTER) ---
        benchmark_strategy(engine, "4. Bilinear (OpenGL Blit)", "Hardware 2D FBO Blitter (`glBlitFramebuffer`, Bilinear)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::Blit),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_BLIT_FILE),

        benchmark_strategy(engine, "5. Bilinear (OpenGL Raster)", "Offscreen Quad Shader (`glDrawArrays`, Bilinear)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::Raster),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_RASTER_FILE),

        benchmark_strategy(engine, "6. Bilinear (OpenGL Compute)", "Direct Compute Shader Resizer (`glDispatchCompute`, Bilinear)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_COMPUTE_FILE),

        // --- 3. HIERARCHICAL MIPMAP / LOD PYRAMID DOWNSCALING ---
        benchmark_strategy(engine, "7. Lod Mipmap (OpenGL Full)", "Hierarchical Multi-Pass Mipchain Reduction (Full Depth)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::LodPyramid, 0),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_LOD_AUTO_FILE),

        benchmark_strategy(engine, "8. Lod Mipmap (OpenGL 2-Pass)", "Hierarchical Multi-Pass Mipchain Reduction (2-Pass Box Limit)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::LodPyramid, 2),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BILINEAR_LOD_2LVL_FILE),

        // --- 4. BICUBIC SPLINE (CATMULL-ROM 4x4) ---
        benchmark_strategy(engine, "9. Bicubic (OpenGL Compute)", "Direct 4x4 Spline Compute Shader (`glDispatchCompute`, Bicubic)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bicubic, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_BICUBIC_COMPUTE_FILE),

        // --- 5. LANCZOS-3 (3-LOBE SINC 6x6) ---
        benchmark_strategy(engine, "10. Lanczos-3 (OpenGL Compute)", "Direct 6x6 Windowed Sinc Compute (`glDispatchCompute`, Lanczos3)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Lanczos3, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_LANCZOS3_COMPUTE_FILE),

        // --- 6. AREA (BOX AVERAGE / OVERLAP) ---
        benchmark_strategy(engine, "11. Area (OpenGL Raster)", "Offscreen Area Box Averaging Quad Shader (`glDrawArrays`, Area)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Area, scalix::Strategy::Raster),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AREA_RASTER_FILE),

        benchmark_strategy(engine, "12. Area (OpenGL Compute)", "Direct Area Box Averaging Compute Shader (`glDispatchCompute`, Area)",
            scalix::ResizeOptions::with_gl(scalix::Filter::Area, scalix::Strategy::Compute),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AREA_COMPUTE_FILE),

        // --- 7. ADAPTIVE ENGINE SELECTOR ---
        benchmark_strategy(engine, "13. Auto (OpenGL Adaptive)", "Scalix Engine Adaptive Fast-Path Strategy Selector",
            scalix::ResizeOptions::with_gl(scalix::Filter::Bilinear, scalix::Strategy::Auto),
            sample.rgb_data, sample.width, sample.height, sample.dst_width, sample.dst_height, num_sample_rounds, OUT_AUTO_FILE),
    };

    // Render Dedicated Results Tables
    TableReporter::print_gl_strategy_table(
        "DEDICATED RESULTS: ALL OPENGL METHODS ON " + sample.path + " (" + to_string(sample.width) + "x" + to_string(sample.height) + " → " + to_string(sample.dst_width) + "x" + to_string(sample.dst_height) + ")",
        sample_benchmarks
    );

    TableReporter::print_profiling_breakdown_table(
        "STAGE-BY-STAGE GPU LATENCY PROFILING BREAKDOWN (" + sample.path + ")",
        sample_benchmarks,
        "OpenGL Hardware Strategy"
    );

    cout << "  Verification artifacts saved to /tmp/bench_gl_sample_*.jpg for visual fidelity inspection." << endl;

#if SCALIX_HAS_OPENCV
    run_opencv_comparison_suite(engine, sample, "OpenGL", [](scalix::Filter filter) {
        if (filter == scalix::Filter::Area) {
            return scalix::ResizeOptions::with_gl(scalix::Filter::Area, scalix::Strategy::Raster);
        }
        return scalix::ResizeOptions::with_gl(filter, scalix::Strategy::Compute);
    });
#endif

    // ========================================================================
    // SECTION 2: SYNTHETIC MULTI-RESOLUTION SCALING BENCHMARKS (RGB888 → 320x320)
    // ========================================================================
    TableReporter::print_section("SECTION 2: MULTI-RESOLUTION DOWNSCALING TO TENSOR (320x320, RGB888)");
    run_synthetic_benchmarks_suite(engine, DEFAULT_BENCH_ROUNDS, "MULTI-RESOLUTION RESOLUTION SCALING THROUGHPUT SUMMARY (OPENGL)");

    cout << "\n[OpenGL benchmark suite completed successfully!]" << endl;
    return 0;
}
