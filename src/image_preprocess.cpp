#include "image_preprocess.h"

#include "stb_image.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>

namespace imgproc {

void StbiFree::operator()(unsigned char* p) const { stbi_image_free(p); }

ImageFormat sniff_image_format(std::string_view b) {
    static const unsigned char kPng[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (b.size() >= 3 && static_cast<unsigned char>(b[0]) == 0xFF &&
        static_cast<unsigned char>(b[1]) == 0xD8 && static_cast<unsigned char>(b[2]) == 0xFF) {
        return ImageFormat::Jpeg;
    }
    if (b.size() >= 8 && std::memcmp(b.data(), kPng, 8) == 0) return ImageFormat::Png;
    return ImageFormat::Unknown;
}

bool strict_base64_decode(std::string_view in, std::string& out) {
    auto value = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t quad[4];
    size_t n = 0;
    size_t pad = 0;
    bool finished = false;
    for (unsigned char c : in) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') continue;
        if (finished) return false;
        if (c == '=') {
            if (n < 2) return false;
            ++pad;
            quad[n++] = 0;
        } else {
            const int v = value(c);
            if (v < 0 || pad) return false;
            quad[n++] = static_cast<uint32_t>(v);
        }
        if (n < 4) continue;
        const uint32_t x = (quad[0] << 18) | (quad[1] << 12) | (quad[2] << 6) | quad[3];
        out.push_back(static_cast<char>((x >> 16) & 0xFF));
        if (pad < 2) out.push_back(static_cast<char>((x >> 8) & 0xFF));
        if (pad < 1) out.push_back(static_cast<char>(x & 0xFF));
        n = 0;
        finished = pad > 0;
    }
    return n == 0 && !out.empty();
}

namespace {

std::string stb_reason() {
    const char* r = stbi_failure_reason();
    return r ? r : "unknown error";
}

}  // namespace

DecodeResult decode_rgb8(std::string_view bytes, const DecodeLimits& limits) {
    DecodeResult r;
    const ImageFormat fmt = sniff_image_format(bytes);
    if (fmt == ImageFormat::Unknown) {
        r.error = DecodeError::NotAnImage;
        r.message = "unsupported image format (JPEG or PNG)";
        return r;
    }
    if (bytes.size() > static_cast<size_t>(INT_MAX)) {
        r.error = DecodeError::NotAnImage;
        r.message = "image is too large";
        return r;
    }
    const auto* data = reinterpret_cast<const stbi_uc*>(bytes.data());
    const int len = static_cast<int>(bytes.size());

    int w = 0, h = 0, comp = 0;
    {
        stb_budget::DecodeBudgetScope info_budget(1u << 20);
        if (!stbi_info_from_memory(data, len, &w, &h, &comp)) {
            // stbi_info tries every loader, so its own reason is only ever
            // "unknown image type"; the sniffed format says more.
            r.error = DecodeError::Corrupt;
            r.message = fmt == ImageFormat::Jpeg
                            ? "cannot read the JPEG header (corrupt or truncated; 12-bit and "
                              "arithmetic-coded JPEGs are not supported)"
                            : "cannot read the PNG header (corrupt or truncated)";
            return r;
        }
    }
    const uint64_t pixels = static_cast<uint64_t>(w) * static_cast<uint64_t>(h);
    if (w <= 0 || h <= 0 || pixels > limits.max_pixels) {
        r.error = DecodeError::TooManyPixels;
        r.message = "image has " + std::to_string(w) + "x" + std::to_string(h) +
                    " pixels; the limit is " + std::to_string(limits.max_pixels);
        return r;
    }

    r.budget_limit = limits.budget_factor * pixels + limits.budget_slack +
                     (fmt == ImageFormat::Png ? 2 * static_cast<uint64_t>(bytes.size()) : 0);
    r.budget_limit = std::min(r.budget_limit, limits.max_budget);
    stb_budget::DecodeBudgetScope budget(r.budget_limit);
    int x = 0, y = 0, n = 0;
    unsigned char* px = stbi_load_from_memory(data, len, &x, &y, &n, 3);
    r.budget_peak = budget.peak();
    if (!px) {
        if (budget.exceeded()) {
            r.error = DecodeError::BudgetExceeded;
            r.message = "image decode exceeded memory budget";
        } else {
            r.error = DecodeError::Corrupt;
            r.message = "cannot decode image: " + stb_reason();
        }
        return r;
    }
    r.image.width = x;
    r.image.height = y;
    r.image.pixels.reset(px);
    return r;
}

namespace {

struct Taps {
    std::vector<int> first;       // first source index per output sample
    std::vector<int> count;       // number of taps per output sample
    std::vector<double> weights;  // `stride` weights per output sample
    int stride = 0;
};

// Pillow's precompute_coeffs with the bilinear filter (support 1), so the
// sample positions and weights match Image.resize(..., Image.BILINEAR).
Taps precompute_taps(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 1.0 * filterscale;
    Taps t;
    t.stride = static_cast<int>(std::ceil(support)) * 2 + 1;
    t.first.resize(out_size);
    t.count.resize(out_size);
    t.weights.assign(static_cast<size_t>(out_size) * t.stride, 0.0);
    const double ss = 1.0 / filterscale;
    auto tri = [](double x) {
        x = std::fabs(x);
        return x < 1.0 ? 1.0 - x : 0.0;
    };
    for (int o = 0; o < out_size; ++o) {
        const double center = (o + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        const int n = std::min(xmax - xmin, t.stride);
        double* w = &t.weights[static_cast<size_t>(o) * t.stride];
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            w[k] = tri((k + xmin - center + 0.5) * ss);
            sum += w[k];
        }
        if (sum != 0.0) {
            for (int k = 0; k < n; ++k) w[k] /= sum;
        }
        t.first[o] = xmin;
        t.count[o] = n;
    }
    return t;
}

}  // namespace

namespace {

// Pillow's 8-bit path: weights in 22-bit fixed point, rounded to uint8 after
// each pass. Emulating it (rather than resizing in float) is what keeps the
// classifier's scores next to the Pillow-preprocessed reference.
constexpr int kPrecisionBits = 32 - 8 - 2;

std::vector<int32_t> fixed_point(const Taps& t) {
    std::vector<int32_t> k(t.weights.size());
    for (size_t i = 0; i < k.size(); ++i) {
        const double w = t.weights[i] * (1 << kPrecisionBits);
        k[i] = static_cast<int32_t>(w < 0 ? w - 0.5 : w + 0.5);
    }
    return k;
}

inline unsigned char clip8(int32_t acc) {
    const int32_t v = acc >> kPrecisionBits;
    return static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

}  // namespace

std::vector<float> resample_triangle(const unsigned char* src, int src_w, int src_h, int channels,
                                     int dst_w, int dst_h) {
    const Taps hx = precompute_taps(src_w, dst_w);
    const Taps vy = precompute_taps(src_h, dst_h);
    const std::vector<int32_t> hk = fixed_point(hx);
    const std::vector<int32_t> vk = fixed_point(vy);
    const size_t c = static_cast<size_t>(channels);
    constexpr int32_t kHalf = 1 << (kPrecisionBits - 1);

    std::vector<unsigned char> tmp(static_cast<size_t>(src_h) * dst_w * c);
    for (int y = 0; y < src_h; ++y) {
        const unsigned char* row = src + static_cast<size_t>(y) * src_w * c;
        unsigned char* out = &tmp[static_cast<size_t>(y) * dst_w * c];
        for (int x = 0; x < dst_w; ++x) {
            const int32_t* k = &hk[static_cast<size_t>(x) * hx.stride];
            const unsigned char* first = row + static_cast<size_t>(hx.first[x]) * c;
            for (size_t ch = 0; ch < c; ++ch) {
                int32_t acc = kHalf;
                for (int i = 0; i < hx.count[x]; ++i) acc += first[static_cast<size_t>(i) * c + ch] * k[i];
                out[static_cast<size_t>(x) * c + ch] = clip8(acc);
            }
        }
    }

    const size_t row_len = static_cast<size_t>(dst_w) * c;
    std::vector<float> dst(static_cast<size_t>(dst_h) * row_len);
    for (int y = 0; y < dst_h; ++y) {
        const int32_t* k = &vk[static_cast<size_t>(y) * vy.stride];
        const unsigned char* first = &tmp[static_cast<size_t>(vy.first[y]) * row_len];
        float* out = &dst[static_cast<size_t>(y) * row_len];
        for (size_t i = 0; i < row_len; ++i) {
            int32_t acc = kHalf;
            for (int r = 0; r < vy.count[y]; ++r) acc += first[static_cast<size_t>(r) * row_len + i] * k[r];
            out[i] = clip8(acc);
        }
    }
    return dst;
}

void normalize_in_place(std::vector<float>& hwc, const std::array<float, 3>& mean,
                        const std::array<float, 3>& std) {
    for (size_t i = 0; i < hwc.size(); ++i) {
        const size_t ch = i % 3;
        hwc[i] = (hwc[i] - mean[ch]) / std[ch];
    }
}

}  // namespace imgproc
