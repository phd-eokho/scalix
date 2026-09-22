#include "jpeg_io.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <cstdio>
#include <csetjmp>
#include <jpeglib.h>

using namespace std;

namespace scalix::examples {

namespace {

constexpr string_view RB_MODE = "rb";
constexpr string_view WB_MODE = "wb";
constexpr string_view LIBJPEG_ERR_PREFIX = "libjpeg Error: ";

struct JpegErrorContext final {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;

    static void error_exit_callback(j_common_ptr cinfo) {
        auto* myerr = reinterpret_cast<JpegErrorContext*>(cinfo->err);
        char buffer[JMSG_LENGTH_MAX];
        (*cinfo->err->format_message)(cinfo, buffer);
        cerr << LIBJPEG_ERR_PREFIX << buffer << endl;
        longjmp(myerr->setjmp_buffer, 1);
    }
};

bool decode_internal(
    string_view filename,
    uint8_t* dst_ptr,
    uint32_t height,
    size_t stride,
    J_COLOR_SPACE color_space
) {
    FILE* infile = fopen(string(filename).c_str(), RB_MODE.data());
    if (!infile) {
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    JpegErrorContext jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = JpegErrorContext::error_exit_callback;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = color_space;
    jpeg_start_decompress(&cinfo);

    while (cinfo.output_scanline < height) {
        JSAMPROW row_pointer = dst_ptr + (cinfo.output_scanline * stride);
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

bool encode_internal(
    string_view filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    size_t stride,
    int num_components,
    J_COLOR_SPACE in_color_space,
    int quality
) {
    FILE* outfile = fopen(string(filename).c_str(), WB_MODE.data());
    if (!outfile) {
        return false;
    }

    struct jpeg_compress_struct cinfo;
    JpegErrorContext jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = JpegErrorContext::error_exit_callback;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        return false;
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = num_components;
    cinfo.in_color_space = in_color_space;

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

} // anonymous namespace

bool JpegIO::read_header(string_view filename, JpegHeader& out_header) {
    FILE* infile = fopen(string(filename).c_str(), RB_MODE.data());
    if (!infile) {
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    JpegErrorContext jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = JpegErrorContext::error_exit_callback;

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

bool JpegIO::decode_rgb888(
    string_view filename,
    uint8_t* dst_ptr,
    uint32_t height,
    size_t stride
) {
    return decode_internal(filename, dst_ptr, height, stride, JCS_RGB);
}

bool JpegIO::decode_rgba8888(
    string_view filename,
    uint8_t* dst_ptr,
    uint32_t height,
    size_t stride
) {
    return decode_internal(filename, dst_ptr, height, stride, JCS_EXT_RGBA);
}

bool JpegIO::encode_rgb888(
    string_view filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    size_t stride,
    int quality
) {
    return encode_internal(filename, width, height, pixels, stride, 3, JCS_RGB, quality);
}

bool JpegIO::encode_rgba8888(
    string_view filename,
    uint32_t width,
    uint32_t height,
    const uint8_t* pixels,
    size_t stride,
    int quality
) {
    return encode_internal(filename, width, height, pixels, stride, 4, JCS_EXT_RGBA, quality);
}

} // namespace scalix::examples
