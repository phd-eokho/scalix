#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <scalix/scalix.hpp>

namespace scalix::examples {

inline constexpr std::string_view BANNER_DIVIDER  = "==========================================================================================";
inline constexpr std::string_view SECTION_DIVIDER = "------------------------------------------------------------------------------------------";

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

} // namespace scalix::examples
