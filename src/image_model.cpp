#include "image_model.h"

#include "counting_semaphore.h"
#include "errors.h"
#include "image_manifest.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using executorch::extension::Module;
using executorch::runtime::Error;

namespace {

constexpr auto kSlotWait = std::chrono::seconds(30);

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string error_name(Error e) {
    char b[8];
    std::snprintf(b, sizeof b, "%x", static_cast<unsigned>(e));
    return std::string("ExecuTorch error 0x") + b;
}

std::string shape_string(const std::vector<int64_t>& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
}

std::vector<float> hwc_to_chw(const std::vector<float>& hwc, int h, int w) {
    const size_t plane = static_cast<size_t>(h) * static_cast<size_t>(w);
    std::vector<float> chw(hwc.size());
    for (size_t p = 0; p < plane; ++p) {
        for (size_t c = 0; c < 3; ++c) chw[c * plane + p] = hwc[p * 3 + c];
    }
    return chw;
}

}  // namespace

struct ImageModel::Impl {
    ImageManifest manifest;
    std::vector<std::string> labels;
    ImageServeOptions options;
    std::unique_ptr<Module> module;
    executorch::runtime::BackendOptions<1> xnnpack_options;
    executorch::runtime::LoadBackendOptionsMap backend_options;
    std::string method;
    std::vector<float> input;
    executorch::extension::TensorPtr input_t;
    int in_h = 0;
    int in_w = 0;
    CountingSemaphore decode_slots;
    std::mutex run_mutex;

    Impl(const fs::path& dir, int threads, const ImageServeOptions& opts, bool verbose)
        : manifest(load_image_manifest(dir)),
          labels(load_labels(dir / manifest.labels_file)),
          options(opts),
          decode_slots(opts.max_concurrent_decodes) {
        const fs::path pte = dir / "model.pte";
        if (!fs::exists(pte)) throw std::runtime_error("model.pte not found in " + dir.string());
        if (threads == 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(
            static_cast<uint32_t>(threads));

        module = std::make_unique<Module>(pte.string(), Module::LoadMode::Mmap);
        if (Error e = module->load(); e != Error::Ok) {
            throw std::runtime_error("cannot load " + pte.string() + " (" + error_name(e) +
                                     "; missing or corrupt .pte?)");
        }
        auto names = module->method_names();
        if (!names.ok()) throw std::runtime_error("cannot list the methods of " + pte.string());
        if (names->count("forward")) {
            method = "forward";
        } else if (names->size() == 1) {
            method = *names->begin();
        } else {
            throw std::runtime_error(pte.string() + " has " + std::to_string(names->size()) +
                                     " methods and none is named forward");
        }

        auto meta = module->method_meta(method);
        if (!meta.ok()) throw std::runtime_error("cannot read the metadata of method " + method);
        if (meta->num_inputs() != 1) {
            throw std::runtime_error("image models must have exactly one input; this one has " +
                                     std::to_string(meta->num_inputs()));
        }
        if (meta->num_outputs() != 1) {
            throw std::runtime_error("image models must have exactly one output; this one has " +
                                     std::to_string(meta->num_outputs()));
        }
        auto tag = meta->input_tag(0);
        auto info = meta->input_tensor_meta(0);
        if (!tag.ok() || *tag != executorch::runtime::Tag::Tensor || !info.ok()) {
            throw std::runtime_error("model input is not a tensor");
        }
        const auto st = info->scalar_type();
        if (st == executorch::aten::ScalarType::Char || st == executorch::aten::ScalarType::Byte ||
            st == executorch::aten::ScalarType::Short) {
            throw std::runtime_error("quantized-input models not supported yet");
        }
        if (st != executorch::aten::ScalarType::Float) throw std::runtime_error("model input must be float32");
        std::vector<int64_t> dims(info->sizes().begin(), info->sizes().end());
        const bool nchw = manifest.layout == TensorLayout::NCHW;
        const bool rank4 = dims.size() == 4 && dims[0] == 1 &&
                           std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d >= 1; });
        if (!nchw && rank4 && dims[1] == 3 && dims[3] != 3) {
            throw std::runtime_error("model input is " + shape_string(dims) +
                                     " (NCHW); set preprocess.layout to \"NCHW\" in manifest.json");
        }
        if (nchw && rank4 && dims[3] == 3 && dims[1] != 3) {
            throw std::runtime_error("model input is " + shape_string(dims) +
                                     " (NHWC) but manifest.json sets preprocess.layout \"NCHW\"");
        }
        if (!rank4 || (nchw ? dims[1] : dims[3]) != 3) {
            throw std::runtime_error(std::string("model input must be ") +
                                     (nchw ? "[1, 3, height, width]" : "[1, height, width, 3]") + "; it is " +
                                     shape_string(dims));
        }
        in_h = static_cast<int>(nchw ? dims[2] : dims[1]);
        in_w = static_cast<int>(nchw ? dims[3] : dims[2]);

        auto otag = meta->output_tag(0);
        auto out = meta->output_tensor_meta(0);
        if (!otag.ok() || *otag != executorch::runtime::Tag::Tensor || !out.ok()) {
            throw std::runtime_error("model output is not a tensor");
        }
        if (out->scalar_type() != executorch::aten::ScalarType::Float) {
            throw std::runtime_error("model output must be float32");
        }
        size_t n = 1;
        for (auto d : out->sizes()) n *= static_cast<size_t>(std::max<int64_t>(d, 0));
        if (n != labels.size()) {
            throw std::runtime_error("model output has " + std::to_string(n) + " scores but " +
                                     manifest.labels_file + " has " + std::to_string(labels.size()) +
                                     " labels");
        }

        const char* wc_env = std::getenv("ET_SERVER_XNNPACK_WEIGHT_CACHE");
        const bool weight_cache = !(wc_env && std::strcmp(wc_env, "0") == 0);
        xnnpack_options.set_option("weight_cache_enabled", weight_cache);
        backend_options.set_options("XnnpackBackend", xnnpack_options.view());
        if (Error e = module->load_method(method, nullptr, nullptr, &backend_options); e != Error::Ok) {
            throw std::runtime_error("cannot load method " + method + " (" + error_name(e) +
                                     "; a missing operator or backend registration?)");
        }
        input.assign(static_cast<size_t>(3) * in_h * in_w, 0.0f);
        std::vector<executorch::aten::SizesType> shape(dims.begin(), dims.end());
        input_t = executorch::extension::from_blob(input.data(), shape, executorch::aten::ScalarType::Float);
        if (verbose) {
            std::fprintf(stderr,
                         "et-server: image-classification, method %s, input %dx%d %s, %zu labels, %d thread(s), "
                         "%d decode slot(s)\n",
                         method.c_str(), in_w, in_h, nchw ? "NCHW" : "NHWC", labels.size(), threads,
                         opts.max_concurrent_decodes);
        }
    }

    json classify(std::string_view bytes, int top_k) {
        if (bytes.size() > options.max_image_bytes) {
            throw PayloadTooLarge("image is " + std::to_string(bytes.size()) + " bytes; the limit is " +
                                  std::to_string(options.max_image_bytes));
        }
        if (bytes.empty()) throw InvalidInput("image is empty");
        if (imgproc::sniff_image_format(bytes) == imgproc::ImageFormat::Unknown) {
            throw InvalidInput("unsupported image format (JPEG or PNG)");
        }

        double decode_ms = 0, preprocess_ms = 0, inference_ms = 0;
        int src_w = 0, src_h = 0;
        std::vector<float> tensor;
        {
            if (!decode_slots.acquire_for(kSlotWait)) {
                throw Busy("busy: no image decode slot became free within 30 s");
            }
            SlotGuard slot(decode_slots);
            auto t0 = std::chrono::steady_clock::now();
            imgproc::DecodeResult d = imgproc::decode_rgb8(bytes, options.limits);
            decode_ms = ms_since(t0);
            if (d.error != imgproc::DecodeError::None) throw InvalidInput(d.message);
            src_w = d.image.width;
            src_h = d.image.height;
            t0 = std::chrono::steady_clock::now();
            tensor = imgproc::resample_triangle(d.image.pixels.get(), src_w, src_h, 3, in_w, in_h);
            d.image.pixels.reset();
            imgproc::normalize_in_place(tensor, manifest.mean, manifest.std);
            if (manifest.layout == TensorLayout::NCHW) tensor = hwc_to_chw(tensor, in_h, in_w);
            preprocess_ms = ms_since(t0);
        }

        std::vector<float> scores(labels.size());
        {
            std::lock_guard<std::mutex> lock(run_mutex);
            const auto t0 = std::chrono::steady_clock::now();
            if (tensor.size() != input.size()) throw std::runtime_error("cannot write the model input");
            std::copy(tensor.begin(), tensor.end(), input.begin());
            std::vector<float>().swap(tensor);
            auto result = module->execute(method, input_t);
            if (!result.ok()) {
                throw std::runtime_error("ExecuTorch execute(" + method + ") failed: " + error_name(result.error()));
            }
            if (result->empty() || !result->at(0).isTensor()) throw std::runtime_error("cannot read the model output");
            const auto& t = result->at(0).toTensor();
            if (static_cast<size_t>(t.numel()) != scores.size()) throw std::runtime_error("cannot read the model output");
            const float* p = t.const_data_ptr<float>();
            std::copy(p, p + scores.size(), scores.begin());
            inference_ms = ms_since(t0);
        }

        // score_normalization "none": the model must already emit probabilities.
        for (float s : scores) {
            if (!(s >= -1e-6f && s <= 1.0f + 1e-6f)) {
                throw std::runtime_error("model output is not probabilities");
            }
        }

        const size_t n = labels.size();
        size_t k = static_cast<size_t>(top_k > 0 ? top_k : manifest.top_k_default);
        k = std::min(k, n);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                          [&](size_t a, size_t b) {
                              return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
                          });

        json predictions = json::array();
        json by_label = json::object();
        for (size_t i = 0; i < k; ++i) {
            const size_t idx = order[i];
            predictions.push_back({{"index", idx}, {"label", labels[idx]}, {"score", scores[idx]}});
            // ImageNet repeats some names ("crane", "maillot"); keep the higher score.
            if (!by_label.contains(labels[idx])) by_label[labels[idx]] = scores[idx];
        }
        return json{{"predictions", predictions},
                    {"labels", by_label},
                    {"input", {{"width", src_w}, {"height", src_h}}},
                    {"timings",
                     {{"decode_ms", decode_ms}, {"preprocess_ms", preprocess_ms}, {"inference_ms", inference_ms}}}};
    }
};

ImageModel::ImageModel(const fs::path& dir, int threads, const ImageServeOptions& options, bool verbose)
    : impl_(std::make_unique<Impl>(dir, threads, options, verbose)) {}

ImageModel::~ImageModel() = default;

json ImageModel::classify(std::string_view image_bytes, int top_k) {
    return impl_->classify(image_bytes, top_k);
}
