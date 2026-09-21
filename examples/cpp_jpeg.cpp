#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <csetjmp>
#include <memory>
#include <scalix/scalix.hpp>
#include <jpeglib.h>

struct JpegCustomErrorMgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    auto* myerr = reinterpret_cast<JpegCustomErrorMgr*>(cinfo->err);
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    std::cerr << "JPEG Error: " << buffer << std::endl;
    longjmp(myerr->setjmp_buffer, 1);
}

struct JpegHeader {
    uint32_t width{0};
    uint32_t height{0};
    int channels{0};
};

static bool read_jpeg_dimensions(const std::string& filename, JpegHeader& out_header) {
    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
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
    const std::string& filename,
    uint8_t* dst_host_ptr,
    uint32_t width,
    uint32_t height,
    size_t stride
) {
    (void)width;
    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
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
    const std::string& filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    size_t stride,
    int quality = 90
) {
    FILE* outfile = fopen(filename.c_str(), "wb");
    if (!outfile) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
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
    const std::string input_path = (argc > 1) ? argv[1] : "assets/sample.jpg";
    const std::string output_path = (argc > 2) ? argv[2] : "output_sample.jpg";

    std::cout << "[Scalix JPEG + Zero-Copy DMA Test]" << std::endl;
    std::cout << "Probing image metadata: " << input_path << std::endl;

    JpegHeader header;
    if (!read_jpeg_dimensions(input_path, header)) {
        std::cerr << "Failed to parse JPEG header." << std::endl;
        return 1;
    }

    std::cout << "  Image Dimensions: " << header.width << "x" << header.height
              << " (" << header.channels << " channels -> RGBA8888)" << std::endl;

    // 1. Initialize Scalix Engine
    scalix::Engine engine(scalix::Backend::Auto, "jpeg");

    // 2. Attempt Zero-Copy DMA buffer allocation
    std::unique_ptr<scalix::DmaBuffer> src_dma;
    std::unique_ptr<scalix::DmaBuffer> dst_dma;
    bool dma_mode = false;

    try {
        src_dma = std::make_unique<scalix::DmaBuffer>(
            header.width,
            header.height,
            scalix::PixelFormat::Rgba8888
        );
        dst_dma = std::make_unique<scalix::DmaBuffer>(
            header.width,
            header.height,
            scalix::PixelFormat::Rgba8888
        );
        dma_mode = true;
        std::cout << "  [DMA Allocator Active] Allocated hardware DMA buffers (src_fd="
                  << src_dma->fd() << ", dst_fd=" << dst_dma->fd() << ", stride="
                  << src_dma->stride() << " bytes)" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  [Host Notice] Hardware DMA device nodes (/dev/dma_heap, /dev/dri) not accessible on this environment." << std::endl;
        std::cout << "                Falling back to standard host memory buffers (" << e.what() << ")." << std::endl;
        dma_mode = false;
    }

    if (dma_mode) {
        // --- DMA-BUF DIRECT ZERO-COPY PATH ---
        std::cout << "Executing direct decoding into mapped DMA memory (via with_write lambda)..." << std::endl;
        bool decode_ok = src_dma->with_write([&](uint8_t* host_ptr, size_t /*size*/) {
            return read_jpeg_direct(input_path, host_ptr, header.width, header.height, src_dma->stride());
        });

        if (!decode_ok) {
            std::cerr << "Failed to decode JPEG into DMA buffer." << std::endl;
            return 1;
        }

        std::cout << "Executing Scalix engine processing on DMA buffers..." << std::endl;
        auto src_desc = src_dma->as_image_desc();
        auto dst_desc = dst_dma->as_image_desc();
        engine.resize(src_desc, dst_desc, scalix::Filter::Passthrough);
        std::cout << "  Processing completed." << std::endl;

        std::cout << "Saving output image directly from DMA memory to: " << output_path << " (via with_read lambda)..." << std::endl;
        bool write_ok = dst_dma->with_read([&](const uint8_t* host_ptr, size_t /*size*/) {
            return write_jpeg_direct(output_path, header.width, header.height, host_ptr, dst_dma->stride(), 90);
        });

        if (!write_ok) {
            std::cerr << "Failed to write output JPEG from DMA buffer." << std::endl;
            return 1;
        }
    } else {
        // --- STANDARD HOST MEMORY PATH ---
        size_t stride = static_cast<size_t>(header.width) * 4;
        size_t data_len = stride * static_cast<size_t>(header.height);
        std::vector<uint8_t> src_buffer(data_len, 0);
        std::vector<uint8_t> dst_buffer(data_len, 0);

        std::cout << "Decoding JPEG into standard host buffer..." << std::endl;
        if (!read_jpeg_direct(input_path, src_buffer.data(), header.width, header.height, stride)) {
            std::cerr << "Failed to decode JPEG into buffer." << std::endl;
            return 1;
        }

        scalix::ImageDesc src{
            .width = header.width,
            .height = header.height,
            .stride_bytes = stride,
            .format = scalix::PixelFormat::Rgba8888,
            .host_ptr = src_buffer.data(),
            .data_len = data_len,
            .dma_buf_fd = -1,
        };

        scalix::ImageDesc dst{
            .width = header.width,
            .height = header.height,
            .stride_bytes = stride,
            .format = scalix::PixelFormat::Rgba8888,
            .host_ptr = dst_buffer.data(),
            .data_len = data_len,
            .dma_buf_fd = -1,
        };

        std::cout << "Executing Scalix engine processing..." << std::endl;
        engine.resize(src, dst, scalix::Filter::Passthrough);
        std::cout << "  Processing completed." << std::endl;

        std::cout << "Saving output JPEG to: " << output_path << std::endl;
        if (!write_jpeg_direct(output_path, header.width, header.height, dst_buffer.data(), stride, 90)) {
            std::cerr << "Failed to write output JPEG image." << std::endl;
            return 1;
        }
    }

    std::cout << "Successfully saved: " << output_path << std::endl;
    return 0;
}
