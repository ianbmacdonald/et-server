#pragma once

// Image decode and preprocessing for /classify/image. No LiteRT dependency, so
// it builds and tests on its own (TFLITE_SERVER_PREPROCESS_TESTS).
//
// Every buffer is heap allocated: musl's default thread stack is 128 KiB and
// the HTTP worker threads run this code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace imgproc {

enum class ImageFormat { Unknown, Jpeg, Png };

// Magic-byte sniff: JPEG (FF D8 FF) or PNG (the 8-byte signature).
ImageFormat sniff_image_format(std::string_view bytes);

// RFC 4648 base64 with '=' padding. ASCII whitespace is skipped; any other
// character outside the alphabet, bad padding or a length that is not a
// multiple of 4 fails (returns false) instead of stopping early.
bool strict_base64_decode(std::string_view in, std::string& out);

struct DecodeLimits {
    uint64_t max_pixels = 4000000;
    // Decode working-set budget: factor * width * height bytes, plus twice the
    // input size for PNG (stb keeps the IDAT payload while it inflates, growing
    // it by doubling), plus a fixed slack.
    uint64_t budget_factor = 16;
    uint64_t budget_slack = 4u << 20;
    // Absolute ceiling on that budget, whatever the factor and pixel cap say.
    uint64_t max_budget = 256u << 20;
};

enum class DecodeError {
    None,
    NotAnImage,       // not JPEG/PNG, or its header cannot be read
    TooManyPixels,    // refused from the header, before decode
    BudgetExceeded,   // an allocation would pass the decode budget
    Corrupt,          // stb rejected the data (truncated, 12-bit, arithmetic, ...)
};

struct StbiFree {
    void operator()(unsigned char* p) const;
};

// Interleaved 8-bit RGB, owned by the stb allocator.
struct Rgb8Image {
    int width = 0;
    int height = 0;
    std::unique_ptr<unsigned char, StbiFree> pixels;
};

struct DecodeResult {
    Rgb8Image image;
    DecodeError error = DecodeError::None;
    std::string message;          // human readable; includes stb's reason when there is one
    uint64_t budget_limit = 0;    // bytes, 0 when decode never started
    uint64_t budget_peak = 0;     // allocator high-water mark during decode
};

// Header check (pixel cap) first, then a budgeted stb decode to RGB.
// Adobe CMYK/YCCK JPEGs decode (stb converts them to RGB). EXIF orientation is
// not applied.
DecodeResult decode_rgb8(std::string_view bytes, const DecodeLimits& limits);

// Pillow-compatible antialiased bilinear ("triangle") resize of an
// interleaved 8-bit image with `channels` channels, as Image.resize(size,
// Image.BILINEAR) does it: horizontal pass then vertical, fixed-point weights,
// uint8 rounding after each pass. Output values are those uint8 results, as
// float on the 0..255 scale.
std::vector<float> resample_triangle(const unsigned char* src, int src_w, int src_h, int channels,
                                     int dst_w, int dst_h);

// (px - mean[c]) / std[c], NHWC order (the resampled image's own order).
void normalize_in_place(std::vector<float>& hwc, const std::array<float, 3>& mean,
                        const std::array<float, 3>& std);

}  // namespace imgproc

// The budgeted stb allocator (stb_image_impl.cpp). A budget is active only
// inside a DecodeBudgetScope on the current thread; outside one, allocations
// are unbounded and uncounted.
namespace stb_budget {

class DecodeBudgetScope {
public:
    explicit DecodeBudgetScope(uint64_t limit);
    ~DecodeBudgetScope();
    DecodeBudgetScope(const DecodeBudgetScope&) = delete;
    DecodeBudgetScope& operator=(const DecodeBudgetScope&) = delete;
    uint64_t peak() const;
    bool exceeded() const;
};

}  // namespace stb_budget
