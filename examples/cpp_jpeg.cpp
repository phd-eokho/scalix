#include <iostream>
#include <vector>
#include <string>
#include <scalix/scalix.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"

int main(int argc, char** argv) {
    const std::string input_path = (argc > 1) ? argv[1] : "assets/sample.jpg";
    const std::string output_path = (argc > 2) ? argv[2] : "output_sample.jpg";

    std::cout << "[Scalix JPEG Test with stb_image]" << std::endl;
    std::cout << "Loading image: " << input_path << std::endl;

    int width = 0;
    int height = 0;
    int channels = 0;

    // Force 4 channels (RGBA) for predictable buffer layout
    unsigned char* raw_pixels = stbi_load(input_path.c_str(), &width, &height, &channels, 4);
    if (!raw_pixels) {
        std::cerr << "Failed to load image: " << input_path << " (" << stbi_failure_reason() << ")" << std::endl;
        return 1;
    }

    std::cout << "  Loaded: " << width << "x" << height << " (orig channels: " << channels << ", forced to RGBA8888)" << std::endl;

    size_t stride = static_cast<size_t>(width) * 4;
    size_t data_len = stride * static_cast<size_t>(height);

    std::vector<uint8_t> dst_buffer(data_len, 0);

    scalix::ImageDesc src{
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = raw_pixels,
        .data_len = data_len,
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst{
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
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

        std::cout << "Saving output image to: " << output_path << std::endl;
        int write_res = stbi_write_jpg(output_path.c_str(), width, height, 4, dst_buffer.data(), 90);
        if (!write_res) {
            std::cerr << "Failed to write output image." << std::endl;
            stbi_image_free(raw_pixels);
            return 1;
        }

        std::cout << "Successfully saved: " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Scalix error: " << e.what() << std::endl;
        stbi_image_free(raw_pixels);
        return 1;
    }

    stbi_image_free(raw_pixels);
    return 0;
}
