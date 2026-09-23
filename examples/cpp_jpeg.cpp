#include <iostream>
#include <string>
#include <string_view>
#include <memory>
#include <scalix/scalix.hpp>
#include "jpeg_io.hpp"

using namespace std;
using scalix::AlignedVector;
using scalix::examples::JpegIO;
using scalix::examples::JpegHeader;
using scalix::examples::DEFAULT_JPEG_QUALITY;

namespace {

constexpr string_view DEFAULT_INPUT_PATH  = "assets/sample.jpg";
constexpr string_view DEFAULT_OUTPUT_PATH = "output_sample.jpg";
constexpr string_view STRAT_AUTO          = "auto";
constexpr string_view STRAT_BLIT          = "blit";
constexpr string_view STRAT_RASTER        = "raster";
constexpr string_view STRAT_LOD           = "lod";
constexpr string_view STRAT_LOD_PYRAMID   = "lod_pyramid";
constexpr string_view STRAT_COMPUTE       = "compute";

constexpr uint32_t DEFAULT_MAX_MIP_LEVELS = 2;
constexpr uint32_t DOWNSCALE_FACTOR       = 8;

void print_latency_breakdown(const scalix::Engine& engine) {
    if (const auto p = engine.last_profile()) {
        cout << "  [Latency Breakdown]" << endl;
        if (p->host_unpack_ms > 0.0) {
            cout << "    Host Unpack (Layout) : " << p->host_unpack_ms << " ms" << endl;
        }
        cout << "    GPU Staging Upload   : " << p->gpu_upload_ms << " ms" << endl;
        cout << "    GPU Core Scaling     : " << p->gpu_pure_blit_ms << " ms" << endl;
        cout << "    GPU Staging Readback : " << p->gpu_download_ms << " ms" << endl;
        if (p->host_repack_ms > 0.0) {
            cout << "    Host Repack (Layout) : " << p->host_repack_ms << " ms" << endl;
        }
        cout << "    Driver/HW Sync Wait  : " << p->driver_sync_ms << " ms" << endl;
        cout << "    Total Wall-Clock     : " << p->total_wall_ms << " ms" << endl;
    }
}

} // anonymous namespace

int main(int argc, char** argv) {
    const string input_path = (argc > 1) ? argv[1] : string(DEFAULT_INPUT_PATH);
    const string output_path = (argc > 2) ? argv[2] : string(DEFAULT_OUTPUT_PATH);
    const string strategy_arg = (argc > 3) ? argv[3] : string(STRAT_AUTO);
    const uint32_t max_mip_levels = (argc > 4) ? static_cast<uint32_t>(stoul(argv[4])) : DEFAULT_MAX_MIP_LEVELS;

    scalix::Strategy strategy = scalix::Strategy::Auto;
    string strategy_name = "Auto (Adaptive Fast-Path Selection)";
    if (strategy_arg == STRAT_BLIT) {
        strategy = scalix::Strategy::Blit;
        strategy_name = "Blit (Hardware Fixed-Function 2D Blitter)";
    } else if (strategy_arg == STRAT_RASTER) {
        strategy = scalix::Strategy::Raster;
        strategy_name = "Raster (Offscreen Graphics Pipeline)";
    } else if (strategy_arg == STRAT_LOD || strategy_arg == STRAT_LOD_PYRAMID) {
        strategy = scalix::Strategy::LodPyramid;
        strategy_name = "LodPyramid (Hierarchical Mipchain Reduction)";
    } else if (strategy_arg == STRAT_COMPUTE) {
        strategy = scalix::Strategy::Compute;
        strategy_name = "Compute (Direct Fused 24-bit Compute Shader)";
    }

    cout << "[Scalix JPEG Hardware Processing Example]" << endl;
    cout << "Pipeline Strategy: " << strategy_name;
    if (strategy == scalix::Strategy::LodPyramid) {
        cout << " (max_mip_levels=" << (max_mip_levels > 0 ? to_string(max_mip_levels) : "auto") << ")";
    }
    cout << endl;

    // 1. Probe JPEG metadata
    JpegHeader header;
    if (!JpegIO::read_header(input_path, header)) {
        cerr << "Failed to parse JPEG header from: " << input_path << endl;
        return 1;
    }

    const uint32_t dst_width = header.width / DOWNSCALE_FACTOR;
    const uint32_t dst_height = header.height / DOWNSCALE_FACTOR;

    cout << "  Source Dimensions: " << header.width << "x" << header.height << " (RGB888)" << endl;
    cout << "  Target Dimensions: " << dst_width << "x" << dst_height << " (1/8 Downscale, Bilinear)" << endl;

    // 2. Initialize Scalix Engine with profiling enabled
    scalix::Engine engine(scalix::Backend::Auto, "jpeg");
    engine.set_profiling(true);

    const scalix::ResizeOptions resize_options = scalix::ResizeOptions::with_vulkan(
        scalix::Filter::Bilinear,
        strategy,
        max_mip_levels
    );

    // 3. Attempt hardware DMA buffer allocation; fallback to host staging
    unique_ptr<scalix::DmaBuffer> src_dma;
    unique_ptr<scalix::DmaBuffer> dst_dma;
    bool dma_mode = false;

    if (strategy != scalix::Strategy::Compute) {
        try {
            src_dma = make_unique<scalix::DmaBuffer>(header.width, header.height, scalix::PixelFormat::Rgba8888);
            dst_dma = make_unique<scalix::DmaBuffer>(dst_width, dst_height, scalix::PixelFormat::Rgba8888);
            dma_mode = true;
            cout << "  [DMA Allocator Active] Allocated hardware DMA buffers (src_fd="
                 << src_dma->fd() << ", dst_fd=" << dst_dma->fd() << ")" << endl;
        } catch (const exception& e) {
            cout << "  [Host Notice] Hardware DMA unavailable (" << e.what() << "). Using standard host memory." << endl;
            dma_mode = false;
        }
    }

    if (dma_mode) {
        // --- ZERO-COPY DMA PATH ---
        cout << "Decoding JPEG into mapped DMA memory..." << endl;
        const bool decode_ok = src_dma->with_write([&](uint8_t* ptr, size_t /*size*/) {
            return JpegIO::decode_rgba8888(input_path, ptr, header.height, src_dma->stride());
        });
        if (!decode_ok) {
            cerr << "Failed to decode JPEG into DMA buffer." << endl;
            return 1;
        }

        cout << "Executing hardware resize on DMA buffers..." << endl;
        auto src_desc = src_dma->as_image_desc();
        auto dst_desc = dst_dma->as_image_desc();
        engine.resize(src_desc, dst_desc, resize_options);
        print_latency_breakdown(engine);

        cout << "Encoding output JPEG directly from DMA memory to: " << output_path << endl;
        const bool write_ok = dst_dma->with_read([&](const uint8_t* ptr, size_t /*size*/) {
            return JpegIO::encode_rgba8888(output_path, dst_width, dst_height, ptr, dst_dma->stride(), DEFAULT_JPEG_QUALITY);
        });
        if (!write_ok) {
            cerr << "Failed to write output JPEG from DMA memory." << endl;
            return 1;
        }
    } else {
        // --- HOST MEMORY RGB888 PATH ---
        const size_t src_stride = header.rgb888_stride();
        const size_t src_size = header.rgb888_size();
        const size_t dst_stride = static_cast<size_t>(dst_width) * 3;
        const size_t dst_size = dst_stride * dst_height;

        AlignedVector<uint8_t> src_buffer(src_size);
        AlignedVector<uint8_t> dst_buffer(dst_size);

        cout << "Decoding JPEG into host memory (packed RGB888)..." << endl;
        if (!JpegIO::decode_rgb888(input_path, src_buffer.data(), header.height, src_stride)) {
            cerr << "Failed to decode JPEG: " << input_path << endl;
            return 1;
        }

        scalix::ImageDesc src{
            .width = header.width,
            .height = header.height,
            .stride_bytes = src_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = src_buffer.data(),
            .data_len = src_size,
            .dma_buf_fd = -1,
        };

        scalix::ImageDesc dst{
            .width = dst_width,
            .height = dst_height,
            .stride_bytes = dst_stride,
            .format = scalix::PixelFormat::Rgb888,
            .host_ptr = dst_buffer.data(),
            .data_len = dst_size,
            .dma_buf_fd = -1,
        };

        cout << "Executing Scalix engine resize (" << header.width << "x" << header.height
             << " → " << dst_width << "x" << dst_height << ")..." << endl;
        engine.resize(src, dst, resize_options);
        print_latency_breakdown(engine);

        cout << "Encoding output JPEG to: " << output_path << endl;
        if (!JpegIO::encode_rgb888(output_path, dst_width, dst_height, dst_buffer.data(), dst_stride, DEFAULT_JPEG_QUALITY)) {
            cerr << "Failed to encode output JPEG to: " << output_path << endl;
            return 1;
        }
    }

    cout << "Successfully saved: " << output_path << " (" << dst_width << "x" << dst_height << ")" << endl;
    return 0;
}
