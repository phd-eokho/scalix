#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <cstdio>
#include <csetjmp>
#include <memory>
#include <scalix/scalix.hpp>
#include <jpeglib.h>

using namespace std;
using scalix::AlignedVector;

struct JpegCustomErrorMgr final {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    auto* myerr = reinterpret_cast<JpegCustomErrorMgr*>(cinfo->err);
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    cerr << "JPEG Error: " << buffer << endl;
    longjmp(myerr->setjmp_buffer, 1);
}

struct JpegHeader final {
    uint32_t width{0};
    uint32_t height{0};
    int channels{0};
};

static bool read_jpeg_dimensions(string_view filename, JpegHeader& out_header) {
    FILE* infile = fopen(string(filename).c_str(), "rb");
    if (!infile) {
        cerr << "Failed to open input file: " << filename << endl;
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    struct JpegCustomErrorMgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    out_header.width = cinfo.image_width;
    out_header.height = cinfo.image_height;
    out_header.channels = cinfo.num_components;

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

static bool read_jpeg_direct(
    string_view filename,
    uint8_t* dst_host_ptr,
    uint32_t width,
    uint32_t height,
    size_t stride
) {
    (void)width;
    FILE* infile = fopen(string(filename).c_str(), "rb");
    if (!infile) {
        cerr << "Failed to open input file: " << filename << endl;
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    struct JpegCustomErrorMgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_EXT_RGBA;
    jpeg_start_decompress(&cinfo);

    while (cinfo.output_scanline < height) {
        JSAMPROW row_pointer = dst_host_ptr + (cinfo.output_scanline * stride);
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

static bool write_jpeg_direct(
    string_view filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    size_t stride,
    int quality = 90
) {
    FILE* outfile = fopen(string(filename).c_str(), "wb");
    if (!outfile) {
        cerr << "Failed to open output file: " << filename << endl;
        return false;
    }

    struct jpeg_compress_struct cinfo;
    struct JpegCustomErrorMgr jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        return false;
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 4;
    cinfo.in_color_space = JCS_EXT_RGBA;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < height) {
        JSAMPROW row_pointer = const_cast<JSAMPROW>(pixels + (cinfo.next_scanline * stride));
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    return true;
}

int main(int argc, char** argv) {
    const string input_path = (argc > 1) ? argv[1] : "assets/sample.jpg";
    const string output_path = (argc > 2) ? argv[2] : "output_sample.jpg";
    const string strategy_arg = (argc > 3) ? argv[3] : "raster";
    const uint32_t max_mip_levels = (argc > 4) ? static_cast<uint32_t>(stoul(argv[4])) : 2;

    scalix::Strategy strategy = scalix::Strategy::Raster;
    string strategy_name = "Raster (Offscreen Graphics Pipeline)";
    if (strategy_arg == "blit") {
        strategy = scalix::Strategy::Blit;
        strategy_name = "Blit (Hardware Fixed-Function 2D Blitter)";
    } else if (strategy_arg == "lod" || strategy_arg == "lod_pyramid") {
        strategy = scalix::Strategy::LodPyramid;
        strategy_name = "LodPyramid (Hierarchical Mipchain Reduction)";
    } else if (strategy_arg == "compute") {
        strategy = scalix::Strategy::Compute;
        strategy_name = "Compute (Programmable Compute Shader Kernel)";
    } else if (strategy_arg == "auto") {
        strategy = scalix::Strategy::Auto;
        strategy_name = "Auto (Default Selection)";
    }

    cout << "[Scalix JPEG + Zero-Copy DMA Test]" << endl;
    cout << "Pipeline Strategy: " << strategy_name;
    if (strategy == scalix::Strategy::LodPyramid) {
        if (max_mip_levels > 0) {
            cout << " (max_mip_levels=" << max_mip_levels << ")";
        } else {
            cout << " (max_mip_levels=auto/unlimited)";
        }
    }
    cout << endl;
    cout << "Probing image metadata: " << input_path << endl;

    JpegHeader header;
    if (!read_jpeg_dimensions(input_path, header)) {
        cerr << "Failed to parse JPEG header." << endl;
        return 1;
    }

    cout << "  Source Dimensions: " << header.width << "x" << header.height
         << " (" << header.channels << " channels → RGBA8888)" << endl;

    uint32_t dst_width = header.width / 8;
    uint32_t dst_height = header.height / 8;

    cout << "  Target Dimensions (1/8 Downscale): " << dst_width << "x" << dst_height
         << " (Filter: Bilinear)" << endl;

    // 1. Initialize Scalix Engine (General purpose coordinator, agnostic of strategy)
    scalix::Engine engine(scalix::Backend::Auto, "jpeg");
    engine.set_profiling(true);

    scalix::ResizeOptions resize_options{
        .filter = scalix::Filter::Bilinear,
        .vulkan = {
            .strategy = strategy,
            .max_mip_levels = max_mip_levels,
        },
    };

    auto print_metrics = [&](const scalix::Engine& eng) {
        if (auto p = eng.last_profile()) {
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
    };

    // 2. Attempt Zero-Copy DMA buffer allocation
    unique_ptr<scalix::DmaBuffer> src_dma;
    unique_ptr<scalix::DmaBuffer> dst_dma;
    bool dma_mode = false;

    try {
        src_dma = make_unique<scalix::DmaBuffer>(
            header.width,
            header.height,
            scalix::PixelFormat::Rgba8888
        );
        dst_dma = make_unique<scalix::DmaBuffer>(
            dst_width,
            dst_height,
            scalix::PixelFormat::Rgba8888
        );
        dma_mode = true;
        cout << "  [DMA Allocator Active] Allocated hardware DMA buffers (src_fd="
             << src_dma->fd() << ", dst_fd=" << dst_dma->fd() << ", src_stride="
             << src_dma->stride() << " bytes, dst_stride=" << dst_dma->stride()
             << " bytes)" << endl;
    } catch (const exception& e) {
        cout << "  [Host Notice] Hardware DMA device nodes (/dev/dma_heap, /dev/dri) not accessible on this environment." << endl;
        cout << "                Falling back to standard host memory buffers (" << e.what() << ")." << endl;
        dma_mode = false;
    }

    if (dma_mode) {
        // --- DMA-BUF DIRECT ZERO-COPY PATH ---
        cout << "Executing direct decoding into mapped DMA memory (via with_write lambda)..." << endl;
        bool decode_ok = src_dma->with_write([&](uint8_t* host_ptr, size_t /*size*/) {
            return read_jpeg_direct(input_path, host_ptr, header.width, header.height, src_dma->stride());
        });

        if (!decode_ok) {
            cerr << "Failed to decode JPEG into DMA buffer." << endl;
            return 1;
        }

        cout << "Executing Scalix engine 1/8 hardware resize on DMA buffers (dynamic strategy: " << strategy_name << ")..." << endl;
        auto src_desc = src_dma->as_image_desc();
        auto dst_desc = dst_dma->as_image_desc();
        engine.resize(src_desc, dst_desc, resize_options);
        cout << "  1/8 resize completed." << endl;
        print_metrics(engine);

        cout << "Saving output 1/8 image directly from DMA memory to: " << output_path << " (via with_read lambda)..." << endl;
        bool write_ok = dst_dma->with_read([&](const uint8_t* host_ptr, size_t /*size*/) {
            return write_jpeg_direct(output_path, dst_width, dst_height, host_ptr, dst_dma->stride(), 90);
        });

        if (!write_ok) {
            cerr << "Failed to write output JPEG from DMA buffer." << endl;
            return 1;
        }
    } else {
        // --- STANDARD HOST MEMORY PATH ---
        size_t src_stride = static_cast<size_t>(header.width) * 4;
        size_t src_data_len = src_stride * static_cast<size_t>(header.height);
        AlignedVector<uint8_t> src_buffer(src_data_len, 0);

        size_t dst_stride = static_cast<size_t>(dst_width) * 4;
        size_t dst_data_len = dst_stride * static_cast<size_t>(dst_height);
        AlignedVector<uint8_t> dst_buffer(dst_data_len, 0);

        cout << "Decoding JPEG into standard host buffer..." << endl;
        if (!read_jpeg_direct(input_path, src_buffer.data(), header.width, header.height, src_stride)) {
            cerr << "Failed to decode JPEG into buffer." << endl;
            return 1;
        }

        scalix::ImageDesc src{
            .width = header.width,
            .height = header.height,
            .stride_bytes = src_stride,
            .format = scalix::PixelFormat::Rgba8888,
            .host_ptr = src_buffer.data(),
            .data_len = src_data_len,
            .dma_buf_fd = -1,
        };

        scalix::ImageDesc dst{
            .width = dst_width,
            .height = dst_height,
            .stride_bytes = dst_stride,
            .format = scalix::PixelFormat::Rgba8888,
            .host_ptr = dst_buffer.data(),
            .data_len = dst_data_len,
            .dma_buf_fd = -1,
        };

        cout << "Executing Scalix engine 1/8 hardware resize (" << header.width << "x" << header.height
             << " → " << dst_width << "x" << dst_height << ") (dynamic strategy: " << strategy_name << ")..." << endl;
        engine.resize(src, dst, resize_options);
        cout << "  1/8 resize completed." << endl;
        print_metrics(engine);

        cout << "Saving 1/8 resized output JPEG to: " << output_path << endl;
        if (!write_jpeg_direct(output_path, dst_width, dst_height, dst_buffer.data(), dst_stride, 90)) {
            cerr << "Failed to write output JPEG image." << endl;
            return 1;
        }
    }

    cout << "Successfully saved: " << output_path << " (" << dst_width << "x" << dst_height << ")" << endl;
    return 0;
}
