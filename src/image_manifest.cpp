#include "image_manifest.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::runtime_error("manifest." + what);
}

void reject_unknown_keys(const json& j, const std::set<std::string>& allowed, const std::string& where) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!allowed.count(it.key())) fail(where + it.key() + ": unknown key");
    }
}

std::string string_value(const json& j, const std::string& key, const std::string& where) {
    if (!j[key].is_string()) fail(where + key + " must be a string");
    return j[key].get<std::string>();
}

// Accepted values pass; reserved ones fail with "not supported in this build";
// anything else is an unknown value.
std::string choice(const json& j, const std::string& key, const std::string& where,
                   const std::set<std::string>& accepted, const std::set<std::string>& reserved) {
    const std::string v = string_value(j, key, where);
    if (accepted.count(v)) return v;
    if (reserved.count(v)) fail(where + key + " '" + v + "' not supported in this build");
    fail(where + key + " '" + v + "' is not a known value");
}

std::array<float, 3> triple(const json& j, const std::string& key, bool positive) {
    const json& v = j[key];
    std::array<float, 3> out{};
    if (v.is_number()) {
        out.fill(v.get<float>());
    } else if (v.is_array() && v.size() == 3 &&
               std::all_of(v.begin(), v.end(), [](const json& x) { return x.is_number(); })) {
        for (size_t i = 0; i < 3; ++i) out[i] = v[i].get<float>();
    } else {
        fail("preprocess." + key + " must be a number or an array of 3 numbers");
    }
    for (float f : out) {
        if (!std::isfinite(f) || (positive && f <= 0.0f)) {
            fail("preprocess." + key + (positive ? " values must be finite and > 0" : " values must be finite"));
        }
    }
    return out;
}

}  // namespace

ImageManifest parse_image_manifest(const json& j) {
    if (!j.is_object()) fail("json: must be an object");
    reject_unknown_keys(j, {"task", "labels_file", "score_normalization", "preprocess", "top_k_default"}, "");
    ImageManifest m;
    if (!j.contains("task") || !j["task"].is_string() || j["task"] != "image-classification") {
        fail("task must be \"image-classification\"");
    }
    if (j.contains("labels_file")) {
        m.labels_file = string_value(j, "labels_file", "");
        const fs::path p(m.labels_file);
        if (m.labels_file.empty() || p.has_parent_path() || p.is_absolute() || m.labels_file == "." ||
            m.labels_file == "..") {
            fail("labels_file must be a file name inside the model directory");
        }
    }
    if (j.contains("score_normalization")) {
        choice(j, "score_normalization", "", {"none"}, {"softmax", "sigmoid"});
    }
    if (j.contains("top_k_default")) {
        if (!j["top_k_default"].is_number_integer() || j["top_k_default"].get<long long>() < 1 ||
            j["top_k_default"].get<long long>() > 1000000) {
            fail("top_k_default must be an integer from 1 to 1000000");
        }
        m.top_k_default = j["top_k_default"].get<int>();
    }
    if (j.contains("preprocess")) {
        const json& p = j["preprocess"];
        if (!p.is_object()) fail("preprocess must be an object");
        reject_unknown_keys(p, {"resize", "mean", "std", "channel_order", "layout"}, "preprocess.");
        if (p.contains("resize")) choice(p, "resize", "preprocess.", {"stretch"}, {"center_crop", "resize_shorter"});
        if (p.contains("channel_order")) choice(p, "channel_order", "preprocess.", {"RGB"}, {"BGR"});
        if (p.contains("layout") && choice(p, "layout", "preprocess.", {"auto", "NHWC", "NCHW"}, {}) == "NCHW") {
            m.layout = TensorLayout::NCHW;
        }
        if (p.contains("mean")) m.mean = triple(p, "mean", false);
        if (p.contains("std")) m.std = triple(p, "std", true);
    }
    return m;
}

ImageManifest load_image_manifest(const fs::path& dir) {
    const fs::path file = dir / "manifest.json";
    std::error_code ec;
    const auto size = fs::file_size(file, ec);
    if (ec) throw std::runtime_error("cannot read " + file.string());
    if (size > (1u << 20)) throw std::runtime_error(file.string() + " is larger than 1 MiB");
    std::ifstream f(file, std::ios::binary);
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(file.string() + " is not valid JSON: " + e.what());
    }
    return parse_image_manifest(j);
}

std::vector<std::string> load_labels(const fs::path& file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open labels file " + file.string());
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        labels.push_back(line);
    }
    if (!labels.empty() && labels.back().empty()) labels.pop_back();
    if (labels.empty()) throw std::runtime_error("labels file " + file.string() + " is empty");
    return labels;
}
