#pragma once

// manifest.json for an image-classification model directory. The schema
// reserves values a later build will implement (center_crop, BGR,
// softmax/sigmoid); this build rejects them by name instead of ignoring them.

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <vector>

enum class TensorLayout { NHWC, NCHW };

struct ImageManifest {
    std::string labels_file = "labels.txt";
    std::array<float, 3> mean = {127.5f, 127.5f, 127.5f};  // 0..255 scale
    std::array<float, 3> std = {127.5f, 127.5f, 127.5f};
    int top_k_default = 5;
    // preprocess.layout: the model input's layout. "auto" (the default) is NHWC.
    TensorLayout layout = TensorLayout::NHWC;
};

// Throws std::runtime_error naming the offending key or value.
ImageManifest parse_image_manifest(const nlohmann::json& j);

// Reads <dir>/manifest.json (at most 1 MiB) and parses it.
ImageManifest load_image_manifest(const std::filesystem::path& dir);

// One label per line; a trailing newline and '\r' line endings are accepted.
std::vector<std::string> load_labels(const std::filesystem::path& file);
