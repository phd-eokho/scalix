#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <csetjmp>
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

struct LoadedImage {
    uint32_t width{0};
    uint32_t height{0};
    std::vector<uint8_t> pixels;
};

static bool read_jpeg_rgba(const std::string& filename, LoadedImage& out_img) {
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

    // Decode directly into RGBA8888 buffer using libjpeg-turbo color space
    cinfo.out_color_space = JCS_EXT_RGBA;
    jpeg_start_decompress(&cinfo);

    out_img.width = cinfo.output_width;
    out_img.height = cinfo.output_height;
    size_t row_stride = static_cast<size_t>(cinfo.output_width) * cinfo.output_components;
    out_img.pixels.resize(row_stride * cinfo.output_height);

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_pointer = out_img.pixels.data() + (cinfo.output_scanline * row_stride);
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

static bool write_jpeg_rgba(
    const std::string& filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
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

    size_t row_stride = static_cast<size_t>(width) * 4;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_pointer = const_cast<JSAMPROW>(pixels + (cinfo.next_scanline * row_stride));
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

    std::cout << "[Scalix JPEG Test with libjpeg-turbo]" << std::endl;
    std::cout << "Loading image via libjpeg: " << input_path << std::endl;

    LoadedImage src_img;
    if (!read_jpeg_rgba(input_path, src_img)) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    std::cout << "  Loaded: " << src_img.width << "x" << src_img.height << " (RGBA8888)" << std::endl;

    size_t stride = static_cast<size_t>(src_img.width) * 4;
    size_t data_len = src_img.pixels.size();
    std::vector<uint8_t> dst_buffer(data_len, 0);

    scalix::ImageDesc src{
        .width = src_img.width,
        .height = src_img.height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = src_img.pixels.data(),
        .data_len = data_len,
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst{
        .width = src_img.width,
        .height = src_img.height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = dst_buffer.data(),
        .data_len = dst_buffer.size(),
        .dma_buf_fd = -1,
    };

    try {
        std::cout << "Initializing Scalix Engine..." << std::endl;
        scalix::Engine engine(scalix::Backend::Auto);

        std::cout << "Executing sync resize (passthrough)..." << std::endl;
        engine.resize(src, dst, scalix::Filter::Passthrough);
        std::cout << "  Processing completed." << std::endl;

        std::cout << "Saving output JPEG to: " << output_path << std::endl;
        if (!write_jpeg_rgba(output_path, src_img.width, src_img.height, dst_buffer.data(), 90)) {
            std::cerr << "Failed to write output JPEG image." << std::endl;
            return 1;
        }

        std::cout << "Successfully saved: " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Scalix error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
