#include <iostream>
#include <vector>
#include <cassert>
#include <atomic>
#include <chrono>
#include <thread>
#include <scalix/scalix.hpp>

int main() {
    std::cout << "[Scalix C++20 Interface Test]" << std::endl;

    // 1. Initialize Engine
    scalix::Engine engine(scalix::Backend::Auto);

    constexpr uint32_t width = 32;
    constexpr uint32_t height = 32;
    constexpr size_t stride = width * 4;
    std::vector<uint8_t> src_data(stride * height, 0x55);
    std::vector<uint8_t> dst_data(stride * height, 0x00);

    scalix::ImageDesc src{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = src_data.data(),
        .data_len = src_data.size(),
        .dma_buf_fd = -1,
    };

    scalix::ImageDesc dst{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = dst_data.data(),
        .data_len = dst_data.size(),
        .dma_buf_fd = -1,
    };

    // 2. Test Synchronous Passthrough
    std::cout << "Testing synchronous resize (passthrough memory transfer)..." << std::endl;
    engine.resize(src, dst, scalix::Filter::Passthrough);
    assert(dst_data == src_data);
    std::cout << "  → Sync transfer verified successfully." << std::endl;

    // 3. Test Callback Execution
    std::cout << "Testing callback-driven resize..." << std::endl;
    std::atomic<bool> cb_done{false};
    std::vector<uint8_t> dst_cb_data(stride * height, 0x00);
    scalix::ImageDesc dst_cb{
        .width = width,
        .height = height,
        .stride_bytes = stride,
        .format = scalix::PixelFormat::Rgba8888,
        .host_ptr = dst_cb_data.data(),
        .data_len = dst_cb_data.size(),
        .dma_buf_fd = -1,
    };

    engine.resize_callback(
        src,
        dst_cb,
        scalix::Filter::Passthrough,
        [&cb_done](int status) {
            std::cout << "  → Callback fired with status: " << status << std::endl;
            cb_done.store(true);
        }
    );

    while (!cb_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "[All C++20 interface tests passed!]" << std::endl;
    return 0;
}
