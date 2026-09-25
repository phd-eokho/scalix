#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <memory>
#include <optional>
#include <functional>
#include <filesystem>
#include <scalix/scalix.hpp>
#include "jpeg_io.hpp"

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/opencv.hpp>
#define SCALIX_HAS_OPENCV 1
#else
#define SCALIX_HAS_OPENCV 0
#endif

namespace scalix::examples {

inline constexpr std::string_view BANNER_DIVIDER  = "========================================================================================================";
inline constexpr std::string_view SECTION_DIVIDER = "--------------------------------------------------------------------------------------------------------";

inline constexpr std::string_view DEFAULT_SAMPLE_PATH   = "assets/sample.jpg";
inline constexpr std::string_view FALLBACK_SAMPLE_PATH  = "../assets/sample.jpg";

inline constexpr size_t DEFAULT_BENCH_ROUNDS       = 16;
inline constexpr size_t PROFILING_ROUNDS           = 5;
inline constexpr size_t CV_BENCH_ROUNDS            = 10;
inline constexpr uint32_t DOWNSCALE_RATIO          = 8;
inline constexpr uint32_t SYNTH_DST_DIM            = 320;

inline constexpr uint32_t SYNTH_4K_W               = 3840;
inline constexpr uint32_t SYNTH_4K_H               = 2160;
inline constexpr uint32_t SYNTH_FHD_W              = 1920;
inline constexpr uint32_t SYNTH_FHD_H              = 1080;
inline constexpr uint32_t SYNTH_HD_W               = 1280;
inline constexpr uint32_t SYNTH_HD_H               = 720;

inline constexpr std::string_view MSG_DMA_ACTIVE   = "[Hardware DMA Status: Native Zero-Copy DMA Active (DMA-Heap / DRM GEM Dumb)]";
inline constexpr std::string_view MSG_DMA_STAGING  = "[Hardware DMA Status: Host-Memory Staging Mode Active (Hardware DMA-BUF / DRM unavailable in current environment)]";

/// @brief Latency and throughput performance metrics.
struct MetricStats final {
    double total_ms{0.0};
    double avg_ms{0.0};
    double fps{0.0};

    [[nodiscard]] static MetricStats from_duration(double total_duration_ms, size_t count);
};

/// @brief Resolution benchmark measurement record.
struct ResolutionBenchmarkResult final {
    std::string name;
    uint32_t src_w{0};
    uint32_t src_h{0};
    uint32_t dst_w{0};
    uint32_t dst_h{0};
    size_t num_frames{0};
    MetricStats sync{};
    MetricStats async{};
    double speedup{1.0};
};

/// @brief Pipeline strategy benchmark measurement record.
struct StrategyBenchmarkResult final {
    std::string method_name;
    std::string description;
    scalix::Strategy strategy{scalix::Strategy::Auto};
    uint32_t max_mip_levels{0};
    scalix::Filter filter{scalix::Filter::Bilinear};
    MetricStats sync{};
    MetricStats async{};
    double speedup{1.0};
    scalix::ProfileMetrics profile{};
    bool success{false};
    std::string output_file;
};

/// @brief Sample image descriptor containing decoded RGB pixel buffer.
struct SampleImageData final {
    std::string path;
    uint32_t width{0};
    uint32_t height{0};
    uint32_t dst_width{0};
    uint32_t dst_height{0};
    scalix::AlignedVector<uint8_t> rgb_data;
    bool is_synthetic{false};
};

/// @brief Formatted terminal table printer adhering to Single Responsibility Principle (SRP).
class TableReporter final {
public:
    static void print_banner(std::string_view title, std::string_view subtitle = "");
    static void print_section(std::string_view section_title);

    static void print_strategy_table(
        std::string_view title,
        const std::vector<StrategyBenchmarkResult>& results,
        std::string_view method_header = "Pipeline Method"
    );

    static void print_vulkan_strategy_table(
        std::string_view title,
        const std::vector<StrategyBenchmarkResult>& results
    ) {
        print_strategy_table(title, results, "Vulkan Pipeline Method");
    }

    static void print_gl_strategy_table(
        std::string_view title,
        const std::vector<StrategyBenchmarkResult>& results
    ) {
        print_strategy_table(title, results, "OpenGL Pipeline Method");
    }

    static void print_opencl_strategy_table(
        std::string_view title,
        const std::vector<StrategyBenchmarkResult>& results
    ) {
        print_strategy_table(title, results, "OpenCL Pipeline Method");
    }

    static void print_profiling_breakdown_table(
        std::string_view title,
        const std::vector<StrategyBenchmarkResult>& results,
        std::string_view strategy_header = "Hardware Strategy"
    );

    static void print_resolution_summary_table(
        std::string_view title,
        const std::vector<ResolutionBenchmarkResult>& results
    );
};

/// @brief Utility to measure execution time of a callable.
class PreciseTimer final {
public:
    template <typename Func>
    static MetricStats measure(size_t rounds, Func&& func) {
        if (rounds == 0) return {};
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < rounds; ++i) {
            func(i);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return MetricStats::from_duration(total_ms, rounds);
    }
};

/// @brief Probes DMA buffer availability and prints status banner.
void probe_and_print_dma_status();

/// @brief Loads sample JPEG image or generates synthetic pattern if absent.
SampleImageData load_or_generate_sample(
    std::string_view requested_path,
    size_t num_rounds = DEFAULT_BENCH_ROUNDS,
    uint32_t downscale_ratio = DOWNSCALE_RATIO
);

/// @brief Executes synchronous, asynchronous, and profiling benchmarks for a strategy.
StrategyBenchmarkResult benchmark_strategy(
    scalix::Engine& engine,
    std::string_view method_name,
    std::string_view description,
    const scalix::ResizeOptions& options,
    const scalix::AlignedVector<uint8_t>& src_buffer,
    uint32_t src_w,
    uint32_t src_h,
    uint32_t dst_w,
    uint32_t dst_h,
    size_t num_frames,
    std::string_view output_filename = ""
);

/// @brief Runs synthetic multi-resolution benchmarks (4K, 1080p, 720p + DMA) and prints summary.
void run_synthetic_benchmarks_suite(
    scalix::Engine& engine,
    size_t num_frames = DEFAULT_BENCH_ROUNDS,
    std::string_view summary_title = "SYNTHETIC MULTI-RESOLUTION BENCHMARK COMPARISON SUMMARY"
);

#if SCALIX_HAS_OPENCV
/// @brief Runs OpenCV CPU vs Scalix GPU comparison benchmarks and prints comparison table.
void run_opencv_comparison_suite(
    scalix::Engine& engine,
    const SampleImageData& sample,
    std::string_view backend_display_name,
    const std::function<scalix::ResizeOptions(scalix::Filter)>& options_factory,
    size_t num_rounds = CV_BENCH_ROUNDS
);
#endif

} // namespace scalix::examples
