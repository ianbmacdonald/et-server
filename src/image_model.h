#pragma once

#include "image_preprocess.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>

struct ImageServeOptions {
    uint64_t max_image_bytes = 16u << 20;
    imgproc::DecodeLimits limits;
    int max_concurrent_decodes = 1;
};

// Image classification: model.pte, one method (float32 [1,H,W,3], or
// [1,3,H,W] with preprocess.layout "NCHW", in; float32 [1,N] probabilities
// out), labels.txt and manifest.json (see image_manifest.h).
class ImageModel {
public:
    ImageModel(const std::filesystem::path& dir, int threads, const ImageServeOptions& options, bool verbose);
    ~ImageModel();
    ImageModel(const ImageModel&) = delete;
    ImageModel& operator=(const ImageModel&) = delete;

    // top_k <= 0 selects the manifest's top_k_default; the result is clamped to
    // the label count. Throws InvalidInput, PayloadTooLarge or Busy (errors.h).
    nlohmann::json classify(std::string_view image_bytes, int top_k);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
