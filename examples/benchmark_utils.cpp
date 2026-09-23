#include "benchmark_utils.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>

using namespace std;

namespace scalix::examples {

namespace {

constexpr int WIDTH_METHOD_NAME = 30;
constexpr int WIDTH_RES_NAME    = 24;
constexpr int WIDTH_STRAT_NAME  = 28;
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

} // namespace scalix::examples
