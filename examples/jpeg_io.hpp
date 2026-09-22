#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>

namespace scalix::examples {

inline constexpr int DEFAULT_JPEG_QUALITY = 90;
inline constexpr int VERIFY_JPEG_QUALITY  = 92;

/// @brief Metadata header for JPEG images.
struct JpegHeader final {
    uint32_t width{0};
    uint32_t height{0};
    int channels{0};

    [[nodiscard]] constexpr size_t rgb888_stride() const noexcept { return static_cast<size_t>(width) * 3; }
    [[nodiscard]] constexpr size_t rgb888_size() const noexcept { return rgb888_stride() * height; }
    [[nodiscard]] constexpr size_t rgba8888_stride() const noexcept { return static_cast<size_t>(width) * 4; }
    [[nodiscard]] constexpr size_t rgba8888_size() const noexcept { return rgba8888_stride() * height; }
};

/// @brief Clean, encapsulated JPEG decoder and encoder adhering to Single Responsibility Principle (SRP).
class JpegIO final {
public:
    /// @brief Probes JPEG metadata header without decoding pixel payload.
    [[nodiscard]] static bool read_header(std::string_view filename, JpegHeader& out_header);

    /// @brief Decodes JPEG image directly into a 24-bit packed RGB888 buffer.
    [[nodiscard]] static bool decode_rgb888(
        std::string_view filename,
        uint8_t* dst_ptr,
        uint32_t height,
        size_t stride
    );

    /// @brief Decodes JPEG image directly into a 32-bit packed RGBA8888 buffer.
    [[nodiscard]] static bool decode_rgba8888(
        std::string_view filename,
        uint8_t* dst_ptr,
        uint32_t height,
        size_t stride
    );

    /// @brief Encodes a 24-bit packed RGB888 pixel buffer into a JPEG file.
    [[nodiscard]] static bool encode_rgb888(
        std::string_view filename,
        uint32_t width,
        uint32_t height,
        const uint8_t* pixels,
        size_t stride,
        int quality = DEFAULT_JPEG_QUALITY
    );

    /// @brief Encodes a 32-bit packed RGBA8888 pixel buffer into a JPEG file.
    [[nodiscard]] static bool encode_rgba8888(
        std::string_view filename,
        uint32_t width,
        uint32_t height,
        const uint8_t* pixels,
        size_t stride,
        int quality = DEFAULT_JPEG_QUALITY
    );
};

} // namespace scalix::examples
