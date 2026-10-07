// et-server — ExecuTorch text-classification server for Lemonade (see README).
//
// Derived from tflite-server, itself derived from lemonade-sdk/ort-server
// (Apache-2.0): the same /classify contract, manifest handling, tokenizer
// handling and model-family allowlist, with the inference engine replaced by an
// ExecuTorch Module (XNNPACK delegate, optimized CPU kernels).
//
// A manifest.json with "task": "image-classification" selects the image path
// instead (POST /classify/image, image_model.cpp, tflite-server's contract).
//
// The text model is a .pte program with one method per fixed sequence length
// (seq_64, seq_128, ...). Each method takes input_ids and attention_mask as
// int64 [1, L] and returns float logits [1, num_labels]. A request runs on the
// smallest method that holds it, padded with a zero attention mask.

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/platform.h>

// tokenizers-cpp's C API: the C++ wrapper hardcodes add_special_tokens=false,
// but encoder classifiers need [CLS]/[SEP] to match the HuggingFace reference.
#include "tokenizers_c.h"

#include "counting_semaphore.h"
#include "errors.h"
#include "flat_json.h"
#include "image_model.h"
#include "image_preprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;

namespace {

// ExecuTorch stores string backend options in a 256-byte array and silently
// truncates longer values, so a cache path near that long would name another file.
constexpr size_t kMaxWeightCachePath = executorch::runtime::kMaxOptionValueLength - 2;
// Bounds the tokenizer's work per request. The whole body is tokenized: a
// byte-level pre-clip would drop text that padding the normalizer discards
// (whitespace, zero-width characters) pushed past the clip, hiding it from the
// classifier even though it falls inside the token window.
constexpr size_t kPayloadMax = 64 * 1024;
constexpr int kOomScoreAdj = 500;
constexpr long long kMaxTopK = 1000000;
constexpr auto kAdmissionWait = std::chrono::seconds(30);
constexpr auto kSocketTimeout = std::chrono::seconds(5);

bool g_verbose = false;

// ExecuTorch logs to stderr by default. A failed start must print exactly one
// "et-server:" line, so runtime logs appear only with --verbose.
void emit_et_log(et_timestamp_t, et_pal_log_level_t level, const char*, const char*, size_t,
                 const char* message, size_t length) {
    if (!g_verbose) return;
    fprintf(stderr, "et-server: [executorch %c] %.*s\n", static_cast<char>(level), static_cast<int>(length), message);
}

struct Manifest {
    std::string task;
    std::vector<std::string> id2label;
    std::string score_normalization = "softmax";
    int max_length = 512;
    // Printed only after a successful start, which must otherwise stay silent
    // on success and print exactly one line on failure.
    std::string deferred_warning;
};

struct Args {
    std::string model_path;
    int port = 0;
    int threads = 0;
    std::string weight_cache;
    std::vector<int> seq_lens;
    bool verbose = false;
    uint64_t max_image_bytes = 16u << 20;
    uint64_t max_image_pixels = 4000000;
    uint64_t decode_budget_factor = 16;
    uint64_t max_decode_bytes = 256u << 20;
    int max_concurrent_decodes = 1;
    int http_threads = 4;
};

std::vector<int> parse_seq_lens(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t pos = 0;
        int n = 0;
        try {
            n = std::stoi(item, &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos == 0 || pos != item.size() || n < 2) {
            throw std::runtime_error("--seq-lens expects a comma-separated list of lengths, got '" + s + "'");
        }
        out.push_back(n);
    }
    if (out.empty()) throw std::runtime_error("--seq-lens is empty");
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

constexpr const char* kUsage =
    "usage: et-server --model-path <dir> --port <n> [--threads N] [--weight-cache FILE] "
    "[--seq-lens 64,512] [--verbose]; image models: [--max-image-bytes N] [--max-image-pixels N] "
    "[--max-concurrent-decodes 1..2] [--decode-budget-factor N] [--max-decode-bytes N] [--http-threads 2..16]";

long long parse_range_flag(const std::string& flag, const std::string& value, long long lo, long long hi) {
    size_t pos = 0;
    long long n = 0;
    try {
        n = std::stoll(value, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos == 0 || pos != value.size() || n < lo || n > hi) {
        throw std::runtime_error(flag + " expects an integer in " + std::to_string(lo) + ".." + std::to_string(hi) +
                                 ", got '" + value + "'");
    }
    return n;
}

int parse_int_flag(const std::string& flag, const std::string& value, int lo, int hi) {
    return static_cast<int>(parse_range_flag(flag, value, lo, hi));
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--verbose") {
            a.verbose = true;
            continue;
        }
        static const std::set<std::string> kValued = {
            "--model-path", "--port", "--threads", "--weight-cache", "--seq-lens", "--max-image-bytes",
            "--max-image-pixels", "--decode-budget-factor", "--max-decode-bytes", "--max-concurrent-decodes",
            "--http-threads"};
        if (!kValued.count(f)) throw std::runtime_error("unknown argument '" + f + "'; " + kUsage);
        if (i + 1 >= argc) throw std::runtime_error(f + " needs a value; " + kUsage);
        const std::string v = argv[++i];
        if (f == "--model-path") a.model_path = v;
        else if (f == "--port") a.port = parse_int_flag(f, v, 1, 65535);
        else if (f == "--threads") a.threads = parse_int_flag(f, v, 1, 1024);
        else if (f == "--weight-cache") a.weight_cache = v;
        else if (f == "--seq-lens") a.seq_lens = parse_seq_lens(v);
        else if (f == "--max-image-bytes") a.max_image_bytes = parse_range_flag(f, v, 1024, 256ll << 20);
        else if (f == "--max-image-pixels") a.max_image_pixels = parse_range_flag(f, v, 1, 16384ll * 16384);
        else if (f == "--decode-budget-factor") a.decode_budget_factor = parse_range_flag(f, v, 4, 64);
        else if (f == "--max-decode-bytes") a.max_decode_bytes = parse_range_flag(f, v, 16ll << 20, 4ll << 30);
        else if (f == "--max-concurrent-decodes") a.max_concurrent_decodes = parse_int_flag(f, v, 1, 2);
        else a.http_threads = parse_int_flag(f, v, 2, 16);
    }
    if (a.model_path.empty() || a.port == 0) throw std::runtime_error(kUsage);
    if (a.weight_cache.size() > kMaxWeightCachePath) {
        throw std::runtime_error("--weight-cache path is " + std::to_string(a.weight_cache.size()) +
                                 " bytes; it must be at most " + std::to_string(kMaxWeightCachePath) +
                                 " (ExecuTorch backend options truncate longer values)");
    }
    return a;
}

// A gateway also runs an LLM; if memory runs out, the classifier should be the
// process the kernel picks first. Only ever raises the score.
void raise_oom_score_adj() {
    std::ifstream in("/proc/self/oom_score_adj");
    int current = 0;
    if (!(in >> current) || current >= kOomScoreAdj) return;
    std::ofstream out("/proc/self/oom_score_adj");
    out << kOomScoreAdj;
}

void parse_id2label(const json& id2label, Manifest& m, const std::string& origin) {
    if (!id2label.is_object() || id2label.empty()) {
        throw std::runtime_error("id2label is missing or empty in " + origin);
    }
    m.id2label.resize(id2label.size());
    std::vector<bool> seen(id2label.size(), false);
    for (auto it = id2label.begin(); it != id2label.end(); ++it) {
        size_t pos = 0;
        unsigned long idx = 0;
        try {
            idx = std::stoul(it.key(), &pos);
        } catch (const std::exception&) {
            pos = 0;
        }
        if (pos != it.key().size()) {
            throw std::runtime_error("id2label key '" + it.key() + "' is not an index in " + origin);
        }
        if (idx >= m.id2label.size() || seen[idx]) {
            throw std::runtime_error("id2label keys must be unique and contiguous 0..n-1 in " + origin);
        }
        if (!it.value().is_string()) {
            throw std::runtime_error("id2label values must be strings in " + origin);
        }
        seen[idx] = true;
        m.id2label[idx] = it.value().get<std::string>();
    }
}

json read_json_if_present(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return json::object();
    try {
        json j;
        f >> j;
        return j.is_object() ? j : json::object();
    } catch (const std::exception&) {
        return json::object();
    }
}

// The server fabricates the attention mask and truncates by keeping the
// trailing token. That is right for BERT-family single-sequence encoders and
// wrong for architectures with other segment/special-token conventions, so the
// supported set is an explicit allowlist.
const std::set<std::string>& supported_model_types() {
    static const std::set<std::string> kSupported = {
        "albert", "bert",     "camembert",  "deberta", "deberta-v2",
        "distilbert", "electra", "roberta", "xlm-roberta", "modernbert",
        "openai_privacy_filter", "pii_masking"
    };
    return kSupported;
}

void validate_model_family(const json& config, const fs::path& dir) {
    if (config.empty()) {
        throw std::runtime_error(
            "config.json is missing or unreadable in " + dir.string() +
            ". It is required (even alongside a manifest.json) to confirm the "
            "model uses the single-sequence encoder convention this server implements.");
    }
    std::string model_type;
    if (config.contains("model_type") && config["model_type"].is_string()) {
        model_type = config["model_type"].get<std::string>();
    }
    if (model_type.empty()) {
        throw std::runtime_error("config.json in " + dir.string() +
                                 " declares no model_type; cannot verify that this "
                                 "architecture uses the single-sequence encoder convention");
    }
    if (!supported_model_types().count(model_type)) {
        std::string supported;
        for (const auto& t : supported_model_types()) supported += (supported.empty() ? "" : ", ") + t;
        throw std::runtime_error("unsupported model_type '" + model_type +
                                 "'. et-server implements the single-sequence encoder "
                                 "convention, which is valid for: " + supported);
    }
}

// The tokenizer's declared budget, then the position table (less 2: RoBERTa
// configs declare more positions than are usable), then 512.
void apply_inferred_max_length(const json& tokenizer_config, const json& config, Manifest& m) {
    auto valid = [](const json& j, const char* key) -> int {
        if (!j.contains(key) || !j[key].is_number_integer()) return 0;
        auto n = j[key].get<long long>();
        return (n >= 2 && n <= 1000000) ? static_cast<int>(n) : 0;
    };
    if (int n = valid(tokenizer_config, "model_max_length")) {
        m.max_length = n;
        return;
    }
    if (int n = valid(config, "max_position_embeddings")) {
        m.max_length = n > 4 ? n - 2 : n;
        return;
    }
    m.max_length = 512;
}

Manifest manifest_from_json(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json");
    if (!f) throw std::runtime_error("cannot open manifest.json in " + dir.string());
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("manifest.json in " + dir.string() + " is not valid JSON: " + e.what());
    }
    Manifest m;
    validate_model_family(read_json_if_present(dir / "config.json"), dir);
    m.task = j.at("task").get<std::string>();
    if (m.task != "text-classification") {
        throw std::runtime_error("unsupported task in manifest.json: '" + m.task +
                                 "' (the text path serves text-classification only; image models set \"task\": \"image-classification\")");
    }
    if (j.contains("score_normalization")) {
        if (!j["score_normalization"].is_string()) {
            throw std::runtime_error("score_normalization must be a string");
        }
        m.score_normalization = j["score_normalization"].get<std::string>();
    }
    if (m.score_normalization != "softmax" && m.score_normalization != "sigmoid") {
        throw std::runtime_error("unsupported score_normalization: " + m.score_normalization);
    }
    if (j.contains("max_length")) {
        if (!j["max_length"].is_number_integer()) throw std::runtime_error("max_length must be an integer");
        m.max_length = j["max_length"].get<int>();
        if (m.max_length < 2) throw std::runtime_error("max_length must be >= 2");
    } else {
        apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"),
                                  read_json_if_present(dir / "config.json"), m);
    }
    parse_id2label(j.at("id2label"), m, "manifest.json");
    return m;
}

// A stock HF export with no manifest.json: infer the contract from config.json.
Manifest manifest_from_hf_config(const fs::path& dir) {
    std::ifstream f(dir / "config.json");
    if (!f) throw std::runtime_error("neither manifest.json nor config.json found in " + dir.string());
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("config.json in " + dir.string() + " is not valid JSON: " + e.what());
    }
    Manifest m;
    validate_model_family(j, dir);
    std::string arch;
    if (j.contains("architectures") && j["architectures"].is_array() && !j["architectures"].empty() &&
        j["architectures"][0].is_string()) {
        arch = j["architectures"][0].get<std::string>();
    }
    const std::string suffix = "ForSequenceClassification";
    if (arch.size() < suffix.size() || arch.compare(arch.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("config.json architecture '" + arch +
                                 "' is not a sequence classifier; provide a manifest.json");
    }
    m.task = "text-classification";
    std::string problem_type;
    if (j.contains("problem_type") && j["problem_type"].is_string()) {
        problem_type = j["problem_type"].get<std::string>();
    }
    if (problem_type == "regression") throw std::runtime_error("regression heads have no label scores in [0,1]");
    if (problem_type == "multi_label_classification") {
        m.score_normalization = "sigmoid";
    } else if (problem_type.empty()) {
        m.deferred_warning =
            "config.json declares no problem_type; assuming SINGLE-LABEL softmax. "
            "A multi-label model needs a manifest.json with \"score_normalization\": \"sigmoid\".";
    }
    parse_id2label(j.at("id2label"), m, "config.json");
    if (m.id2label.size() < 2) throw std::runtime_error("single-output heads have no label scores in [0,1]");
    apply_inferred_max_length(read_json_if_present(dir / "tokenizer_config.json"), j, m);
    return m;
}

Manifest load_manifest(const fs::path& dir) {
    if (fs::exists(dir / "manifest.json")) return manifest_from_json(dir);
    return manifest_from_hf_config(dir);
}

std::vector<float> normalize(const float* v, size_t n, const std::string& mode) {
    std::vector<float> out(n);
    if (mode == "softmax") {
        float mx = *std::max_element(v, v + n);
        double sum = 0;
        for (size_t i = 0; i < n; ++i) { out[i] = std::exp(v[i] - mx); sum += out[i]; }
        for (auto& x : out) x = static_cast<float>(x / sum);
        return out;
    }
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-v[i]));
    return out;
}

std::string load_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Error text can echo raw request bytes (nlohmann's parse errors quote the
// input), and the default strict dump() throws on invalid UTF-8 inside the
// handler, which turns a 400 into httplib's 500.
std::string error_body(const std::string& message) {
    return json{{"error", message}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

size_t default_thread_stack_kib() {
    pthread_attr_t attr;
    size_t size = 0;
    if (pthread_attr_init(&attr) == 0) {
        pthread_attr_getstacksize(&attr, &size);
        pthread_attr_destroy(&attr);
    }
    return size / 1024;
}

std::string error_name(Error e) {
    return "ExecuTorch error 0x" + [&] {
        char b[8];
        snprintf(b, sizeof b, "%x", static_cast<unsigned>(e));
        return std::string(b);
    }();
}

bool parse_method_len(const std::string& name, int& len) {
    if (name.rfind("seq_", 0) != 0 || name.size() == 4) return false;
    for (size_t i = 4; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    len = std::stoi(name.substr(4));
    return len >= 2;
}

class Model {
public:
    Model(const fs::path& dir, const Args& args) : manifest_(load_manifest(dir)), verbose_(args.verbose) {
        std::string blob = load_bytes(dir / "tokenizer.json");
        // The Rust tokenizer unwraps its parse Result, so a truncated
        // tokenizer.json would abort the process with no usable message.
        json tj;
        try {
            tj = json::parse(blob);
        } catch (const std::exception& e) {
            throw std::runtime_error("tokenizer.json in " + dir.string() +
                                     " is not valid JSON (truncated or corrupt download?): " + e.what());
        }
        tokenizer_ = tokenizers_new_from_str(blob.data(), blob.size());
        if (!tokenizer_) throw std::runtime_error("failed to load tokenizer.json from " + dir.string());
        // A tokenizer.json with a padding section pads every encoding; the HF
        // reference does not, so those trailing pad ids are dropped.
        if (tj.contains("padding") && tj["padding"].is_object() && tj["padding"].contains("pad_id") &&
            tj["padding"]["pad_id"].is_number_integer()) {
            pad_id_ = tj["padding"]["pad_id"].get<int64_t>();
        }

        const fs::path pte = dir / "model.pte";
        if (!fs::exists(pte)) throw std::runtime_error("model.pte not found in " + dir.string());

        int threads = args.threads;
        if (threads == 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(
            static_cast<uint32_t>(threads));

        const auto t0 = std::chrono::steady_clock::now();
        // Mmap keeps the .pte file-backed (reclaimable). share_memory_arenas gives
        // one activation arena sized for the longest method instead of one each.
        module_ = std::make_unique<Module>(pte.string(), Module::LoadMode::Mmap, nullptr, nullptr, nullptr,
                                           /*share_memory_arenas=*/true);
        if (Error e = module_->load(); e != Error::Ok) {
            throw std::runtime_error("cannot load " + pte.string() + " (" + error_name(e) +
                                     "; missing or corrupt .pte?)");
        }
        auto names = module_->method_names();
        if (!names.ok()) throw std::runtime_error("cannot list the methods of " + pte.string());
        std::map<int, std::string> available;
        for (const auto& n : *names) {
            int len = 0;
            if (parse_method_len(n, len)) available[len] = n;
        }
        if (available.empty()) throw std::runtime_error(pte.string() + " has no seq_<L> methods");
        std::vector<int> wanted;
        if (args.seq_lens.empty()) {
            for (const auto& [len, _] : available) wanted.push_back(len);
        } else {
            for (int len : args.seq_lens) {
                if (!available.count(len)) {
                    std::string have;
                    for (const auto& [l, _] : available) have += (have.empty() ? "" : ",") + std::to_string(l);
                    throw std::runtime_error("--seq-lens asks for seq_" + std::to_string(len) +
                                             ", which model.pte does not have (it has " + have + ")");
                }
                wanted.push_back(len);
            }
        }

        // The XNNPACK weight cache packs each named constant once and shares it
        // across methods; without it every method holds its own packed copy.
        const char* wc_env = std::getenv("ET_SERVER_XNNPACK_WEIGHT_CACHE");
        const bool weight_cache = !(wc_env && std::strcmp(wc_env, "0") == 0);
        xnnpack_options_.set_option("weight_cache_enabled", weight_cache);
        backend_options_.set_options("XnnpackBackend", xnnpack_options_.view());

        for (int len : wanted) {
            Method m;
            m.len = static_cast<size_t>(len);
            m.name = available[len];
            validate_method(m.name, m.len);
            if (Error e = module_->load_method(m.name, nullptr, nullptr, &backend_options_); e != Error::Ok) {
                throw std::runtime_error("cannot load method " + m.name + " (" + error_name(e) +
                                         "; a missing operator or backend registration?)");
            }
            m.ids.assign(m.len, 0);
            m.mask.assign(m.len, 0);
            m.ids_t = executorch::extension::from_blob(m.ids.data(), {1, len}, executorch::aten::ScalarType::Long);
            m.mask_t = executorch::extension::from_blob(m.mask.data(), {1, len}, executorch::aten::ScalarType::Long);
            methods_.push_back(std::move(m));
        }
        if (static_cast<size_t>(manifest_.max_length) > methods_.back().len) {
            manifest_.max_length = static_cast<int>(methods_.back().len);
        }

        // Warm every method once so the first request pays no first-run cost.
        const std::vector<int64_t> warm_ids = encode("");
        if (warm_ids.empty()) throw std::runtime_error("tokenizer.json encodes the empty string to no tokens");
        for (auto& m : methods_) run(m, warm_ids);
        load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (verbose_) {
            std::string lens;
            for (const auto& m : methods_) lens += (lens.empty() ? "" : ",") + std::to_string(m.len);
            fprintf(stderr,
                    "et-server: %zu method(s), lengths %s, %d thread(s), xnnpack weight cache %s, "
                    "max_length %d, loaded+warmed in %.0f ms\n",
                    methods_.size(), lens.c_str(), threads, weight_cache ? "on" : "off", manifest_.max_length,
                    load_ms_);
            fprintf(stderr, "et-server: default thread stack %zu KiB\n", default_thread_stack_kib());
        }
        if (!args.weight_cache.empty()) {
            fprintf(stderr, "et-server: --weight-cache %s accepted; the on-disk cache is not used in v0.1.0\n",
                    args.weight_cache.c_str());
        }
        if (!manifest_.deferred_warning.empty()) {
            fprintf(stderr, "et-server: %s\n", manifest_.deferred_warning.c_str());
        }
    }

    ~Model() {
        if (tokenizer_) tokenizers_free(tokenizer_);
    }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    json classify(const std::string& text, int top_k) {
        const size_t max_len = static_cast<size_t>(manifest_.max_length);
        std::vector<int64_t> input_ids = encode(text);
        if (input_ids.empty()) throw std::runtime_error("empty tokenization");
        // Keep the trailing token ([SEP] / </s>) so the sequence stays well-formed.
        if (input_ids.size() > max_len) {
            int64_t last = input_ids.back();
            input_ids.resize(max_len - 1);
            input_ids.push_back(last);
        }

        Method* m = &methods_.back();
        for (auto& candidate : methods_) {
            if (candidate.len >= input_ids.size()) { m = &candidate; break; }
        }
        if (verbose_) {
            fprintf(stderr, "et-server: %zu tokens (last id %lld) -> %s\n", input_ids.size(),
                    static_cast<long long>(input_ids.back()), m->name.c_str());
        }
        const std::vector<float> logits = run(*m, input_ids);

        auto p = normalize(logits.data(), logits.size(), manifest_.score_normalization);
        std::vector<std::pair<std::string, float>> ranked;
        for (size_t l = 0; l < p.size(); ++l) ranked.emplace_back(manifest_.id2label[l], p[l]);
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if (top_k > 0 && static_cast<size_t>(top_k) < ranked.size()) ranked.resize(top_k);
        json labels = json::object();
        for (auto& [label, score] : ranked) labels[label] = score;
        return json{{"labels", labels}};
    }

private:
    std::vector<int64_t> encode(const std::string& text) {
        std::vector<int64_t> ids;
        {
            std::lock_guard<std::mutex> lock(tokenizer_mutex_);
            TokenizerEncodeResult result;
            tokenizers_encode(tokenizer_, text.data(), text.size(), /*add_special_token=*/1, &result);
            ids.assign(result.token_ids, result.token_ids + result.len);
            tokenizers_free_encode_results(&result, 1);
        }
        if (pad_id_ >= 0) {
            while (ids.size() > 1 && ids.back() == pad_id_) ids.pop_back();
        }
        return ids;
    }

    struct Method {
        std::string name;
        size_t len = 0;
        std::vector<int64_t> ids;
        std::vector<int64_t> mask;
        TensorPtr ids_t;
        TensorPtr mask_t;
    };

    // v0.1.0 serves exactly the exporter's layout; anything else is refused at
    // startup rather than fed wrong-typed buffers at request time.
    void validate_method(const std::string& name, size_t len) {
        auto meta = module_->method_meta(name);
        if (!meta.ok()) throw std::runtime_error("cannot read the metadata of method " + name);
        auto fail = [&](const std::string& why) {
            throw std::runtime_error("method " + name + ": " + why +
                                     " (v0.1.0 expects inputs input_ids, attention_mask as int64 [1," +
                                     std::to_string(len) + "] and output 0 as float32 [1," +
                                     std::to_string(manifest_.id2label.size()) + "])");
        };
        if (meta->num_inputs() != 2) fail(std::to_string(meta->num_inputs()) + " inputs");
        for (size_t i = 0; i < 2; ++i) {
            auto tag = meta->input_tag(i);
            if (!tag.ok() || *tag != executorch::runtime::Tag::Tensor) fail("input " + std::to_string(i) + " is not a tensor");
            auto info = meta->input_tensor_meta(i);
            if (!info.ok()) fail("input " + std::to_string(i) + " has no tensor metadata");
            if (info->scalar_type() != executorch::aten::ScalarType::Long) {
                fail("input " + std::to_string(i) + " is not int64");
            }
            auto s = info->sizes();
            if (s.size() != 2 || s[0] != 1 || static_cast<size_t>(s[1]) != len) {
                fail("input " + std::to_string(i) + " is not [1," + std::to_string(len) + "]");
            }
        }
        if (meta->num_outputs() < 1) fail("no outputs");
        auto otag = meta->output_tag(0);
        if (!otag.ok() || *otag != executorch::runtime::Tag::Tensor) fail("output 0 is not a tensor");
        auto out = meta->output_tensor_meta(0);
        if (!out.ok()) fail("output 0 has no tensor metadata");
        if (out->scalar_type() != executorch::aten::ScalarType::Float) fail("output 0 is not float32");
        auto os = out->sizes();
        if (os.size() != 2 || os[0] != 1 || static_cast<size_t>(os[1]) != manifest_.id2label.size()) {
            fail("output 0 shape does not match [1, id2label size]");
        }
    }

    // Module is not thread-safe, and with shared arenas one method's output can
    // alias another's activations, so every execute runs under run_mutex_ and
    // the logits are copied out before it is released.
    std::vector<float> run(Method& m, const std::vector<int64_t>& input_ids) {
        std::lock_guard<std::mutex> lock(run_mutex_);
        const int64_t pad = pad_id_ >= 0 ? pad_id_ : 0;
        std::fill(m.ids.begin(), m.ids.end(), pad);
        std::fill(m.mask.begin(), m.mask.end(), 0);
        for (size_t i = 0; i < input_ids.size() && i < m.len; ++i) {
            m.ids[i] = input_ids[i];
            m.mask[i] = 1;
        }
        auto result = module_->execute(m.name, {m.ids_t, m.mask_t});
        if (!result.ok()) throw std::runtime_error("ExecuTorch execute(" + m.name + ") failed: " + error_name(result.error()));
        if (result->empty() || !result->at(0).isTensor()) throw std::runtime_error(m.name + " returned no tensor");
        const auto& t = result->at(0).toTensor();
        const size_t n = static_cast<size_t>(t.numel());
        if (n != manifest_.id2label.size()) {
            throw std::runtime_error("model output has " + std::to_string(n) + " values; manifest id2label has " +
                                     std::to_string(manifest_.id2label.size()));
        }
        const float* p = t.const_data_ptr<float>();
        return std::vector<float>(p, p + n);
    }

    Manifest manifest_;
    bool verbose_ = false;
    std::unique_ptr<Module> module_;
    BackendOptions<1> xnnpack_options_;
    LoadBackendOptionsMap backend_options_;
    std::vector<Method> methods_;
    std::mutex run_mutex_;
    TokenizerHandle tokenizer_ = nullptr;
    int64_t pad_id_ = -1;
    std::mutex tokenizer_mutex_;
    double load_ms_ = 0;
};

// True when <dir>/manifest.json is a JSON object with "task":
// "image-classification". Anything else, including an unreadable manifest, is
// left to the text path, which reports its own errors.
bool is_image_model(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json", std::ios::binary);
    if (!f) return false;
    json j;
    try {
        f >> j;
    } catch (const std::exception&) {
        return false;
    }
    return j.is_object() && j.contains("task") && j["task"] == "image-classification";
}

void send_error(httplib::Response& res, int status, const std::string& message) {
    res.status = status;
    res.set_content(error_body(message), "application/json");
}

// top_k follows Lemonade's /v1/classify rule: an integer from 1 to 1,000,000.
int top_k_from_json(const json& v) {
    if (!v.is_number_integer() || v.get<long long>() < 1 || v.get<long long>() > kMaxTopK) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return static_cast<int>(v.get<long long>());
}

int top_k_from_field(const std::string& s) {
    if (s.empty() || s.size() > 7 || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return top_k_from_json(json(std::stoll(s)));
}

uint64_t content_length(const httplib::Request& req) {
    return req.has_header("Content-Length") ? req.get_header_value_u64("Content-Length") : 0;
}

void drain(const httplib::Request& req, const httplib::ContentReader& reader) {
    if (req.is_multipart_form_data()) {
        reader([](const httplib::MultipartFormData&) { return true; }, [](const char*, size_t) { return true; });
    } else {
        reader([](const char*, size_t) { return true; });
    }
}

// Reads the body through httplib's ContentReader, so nothing is buffered
// before the handler has an admission slot. Receivers never abort: an
// oversize or unwanted part is discarded while the stream is still read to
// its end, which keeps the connection usable.
//
// Multipart: exactly one file part named "image" or "file", plus an optional
// "top_k" field. JSON: {"image": "<base64 or data:image/(jpeg|png);base64,...>", "top_k": k}.
void read_image_request(const httplib::Request& req, const httplib::Response& res,
                        const httplib::ContentReader& reader, uint64_t max_image_bytes, uint64_t max_body_bytes, std::string& bytes, int& top_k) {
    top_k = 0;
    bytes.clear();
    bool too_large = false;
    bool ok = false;
    if (req.is_multipart_form_data()) {
        enum class Part { Image, TopK, Other } cur = Part::Other;
        size_t image_parts = 0;
        std::string top_k_field;
        bool has_top_k = false;
        ok = reader(
            [&](const httplib::MultipartFormData& f) {
                if (f.name == "image" || f.name == "file") {
                    cur = ++image_parts == 1 && !too_large ? Part::Image : Part::Other;
                } else if (f.name == "top_k") {
                    cur = Part::TopK;
                    has_top_k = true;
                    top_k_field.clear();
                } else {
                    cur = Part::Other;
                }
                return true;
            },
            [&](const char* d, size_t n) {
                if (cur == Part::Image) {
                    if (bytes.size() + n > max_image_bytes) {
                        too_large = true;
                        cur = Part::Other;
                        std::string().swap(bytes);
                    } else {
                        bytes.append(d, n);
                    }
                } else if (cur == Part::TopK && top_k_field.size() < 16) {
                    top_k_field.append(d, std::min<size_t>(n, 16));
                }
                return true;
            });
        if (ok && too_large) {
            throw PayloadTooLarge("image is larger than the " + std::to_string(max_image_bytes) + " byte limit");
        }
        if (ok) {
            if (image_parts != 1) throw InvalidInput("exactly one image part ('image' or 'file') required");
            if (has_top_k) top_k = top_k_from_field(top_k_field);
            return;
        }
    } else {
        std::string body;
        const uint64_t declared = content_length(req);
        if (declared <= max_body_bytes) body.reserve(static_cast<size_t>(declared));
        ok = reader([&](const char* d, size_t n) {
            if (too_large) return true;
            if (body.size() + n > max_body_bytes) {
                too_large = true;
                std::string().swap(body);
            } else {
                body.append(d, n);
            }
            return true;
        });
        if (ok && too_large) {
            throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
        }
        if (ok) {
            json obj;
            std::string error;
            if (!parse_flat_json_object(body, obj, error)) {
                if (error == "request body is not valid JSON") error += " (send multipart/form-data or a JSON object)";
                throw InvalidInput(error);
            }
            std::string().swap(body);
            if (!obj.contains("image") || !obj["image"].is_string()) {
                throw InvalidInput("'image' (a base64 string) is required");
            }
            if (obj.contains("top_k")) top_k = top_k_from_json(obj["top_k"]);
            std::string image = std::move(obj["image"].get_ref<std::string&>());
            obj = json();
            std::string_view s = image;
            if (s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0) {
                throw InvalidInput("remote image URLs are not supported; send base64 or a data: URL");
            }
            if (s.rfind("data:", 0) == 0) {
                bool prefix_ok = false;
                for (std::string_view prefix : {"data:image/jpeg;base64,", "data:image/png;base64,"}) {
                    if (s.rfind(prefix, 0) == 0) {
                        s.remove_prefix(prefix.size());
                        prefix_ok = true;
                        break;
                    }
                }
                if (!prefix_ok) throw InvalidInput("data URLs must be data:image/jpeg;base64 or data:image/png;base64");
            }
            if (!imgproc::strict_base64_decode(s, bytes)) throw InvalidInput("'image' is not valid base64");
            return;
        }
    }
    std::string().swap(bytes);
    // A failed read has set res.status: 413 for a Content-Length over the
    // payload limit, 400 for a malformed or truncated body.
    if (res.status == 413) {
        throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
    }
    throw InvalidInput("cannot read the request body (malformed multipart or truncated)");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        executorch::runtime::register_pal(executorch::runtime::PalImpl::create(&emit_et_log, __FILE__));
        Args args = parse_args(argc, argv);
        g_verbose = args.verbose;
        raise_oom_score_adj();
        std::unique_ptr<Model> text_model;
        std::unique_ptr<ImageModel> image;
        if (is_image_model(args.model_path)) {
            ImageServeOptions opts;
            opts.max_image_bytes = args.max_image_bytes;
            opts.limits.max_pixels = args.max_image_pixels;
            opts.limits.budget_factor = args.decode_budget_factor;
            opts.limits.max_budget = args.max_decode_bytes;
            opts.max_concurrent_decodes = args.max_concurrent_decodes;
            image = std::make_unique<ImageModel>(args.model_path, args.threads, opts, args.verbose);
        } else {
            text_model = std::make_unique<Model>(args.model_path, args);
        }

        httplib::Server srv;
        // Per-recv idle limits, not a cap on a request's total time.
        srv.set_read_timeout(kSocketTimeout);
        srv.set_write_timeout(kSocketTimeout);
        // httplib's default sets SO_REUSEPORT on Linux, which lets a second
        // instance share a live port and receive part of its traffic.
        srv.set_socket_options([](socket_t sock) {
            int one = 1;
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        });
        // Base64 inflates by 4/3; 64 KiB covers multipart headers and the JSON wrapper.
        const uint64_t max_body_bytes = args.max_image_bytes * 4 / 3 + (64u << 10);
        // Image requests read their body only after taking one of these, so
        // http_threads bounds waiting connections, not buffered bodies. One
        // more than the decode slots lets the next body arrive during a decode.
        CountingSemaphore admission(args.max_concurrent_decodes + 1);
        if (image) {
            const size_t http_threads = static_cast<size_t>(args.http_threads);
            srv.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };
            srv.set_payload_max_length(static_cast<size_t>(max_body_bytes));
            if (args.decode_budget_factor * args.max_image_pixels + (4u << 20) > args.max_decode_bytes) {
                fprintf(stderr,
                        "et-server: warning: --decode-budget-factor x --max-image-pixels exceeds "
                        "--max-decode-bytes; images near the pixel cap may fail the decode budget\n");
            }
        } else {
            srv.set_payload_max_length(kPayloadMax);
        }
        srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            json health{{"status", "ok"}, {"engine", "executorch"}};
            if (image) health["task"] = "image-classification";
            res.set_content(health.dump(), "application/json");
        });
        srv.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
            if (!text_model) {
                send_error(res, 400, "this server hosts an image-classification model; POST /classify/image");
                return;
            }
            Model& model = *text_model;
            std::string text;
            int top_k = 0;
            try {
                json body = json::parse(req.body);
                text = body.contains("text") ? body.at("text").get<std::string>()
                                             : body.at("input").get<std::string>();
                top_k = body.value("top_k", 0);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(error_body(e.what()), "application/json");
                return;
            }
            try {
                res.set_content(model.classify(text, top_k).dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content(error_body(e.what()), "application/json");
            }
        });
        srv.Post("/classify/image", [&](const httplib::Request& req, httplib::Response& res,
                                        const httplib::ContentReader& reader) {
            if (!image) {
                drain(req, reader);
                send_error(res, 400, "this server hosts a text model; POST /classify");
                return;
            }
            if (content_length(req) > max_body_bytes) {
                drain(req, reader);
                send_error(res, 413, "request body is over the " + std::to_string(max_body_bytes) + " byte limit");
                return;
            }
            if (!admission.acquire_for(kAdmissionWait)) {
                drain(req, reader);
                send_error(res, 503, "busy: no request slot became free within 30 s");
                return;
            }
            SlotGuard slot(admission);
            try {
                std::string bytes;
                int top_k = 0;
                read_image_request(req, res, reader, args.max_image_bytes, max_body_bytes, bytes, top_k);
                res.set_content(image->classify(bytes, top_k).dump(), "application/json");
            } catch (const InvalidInput& e) {
                send_error(res, 400, e.what());
            } catch (const PayloadTooLarge& e) {
                send_error(res, 413, e.what());
            } catch (const Busy& e) {
                send_error(res, 503, e.what());
            } catch (const std::exception& e) {
                send_error(res, 500, e.what());
            }
        });

        if (!srv.listen("127.0.0.1", args.port)) {
            fprintf(stderr, "et-server: failed to bind 127.0.0.1:%d\n", args.port);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "et-server: %s\n", e.what());
        return 1;
    }
}
