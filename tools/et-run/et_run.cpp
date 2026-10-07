// et-run: run one method of a .pte on raw input files and write every tensor output as float32.
// Links exactly what et-server links (same archives, same operator registration library), so it proves
// that a given ExecuTorch build (e.g. a selective one) can run a model et-server does not serve.
//
//   et-run <model.pte> <method> -- <out_prefix> <file>:<dtype>:<d0,d1,...> ... [-- <out_prefix> ...]
//   et-run <model.pte> <method> <input.f32> <d0,d1,...> [output.f32]      (single float input)
//
// dtype: f32 i32 i64 u8 i8 bool; inputs are positional. Output i goes to <out_prefix>.<i>.f32.
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::runtime::EValue;

namespace {

struct Input {
    std::vector<uint8_t> data;
    std::vector<executorch::aten::SizesType> shape;
    ScalarType type;
};

bool parse(const std::string& s, Input& o) {
    auto c2 = s.rfind(':');
    if (c2 == std::string::npos || c2 == 0) return false;
    auto c1 = s.rfind(':', c2 - 1);
    if (c1 == std::string::npos) return false;
    std::string file = s.substr(0, c1), d = s.substr(c1 + 1, c2 - c1 - 1);
    if (d == "f32") o.type = ScalarType::Float;
    else if (d == "i32") o.type = ScalarType::Int;
    else if (d == "i64") o.type = ScalarType::Long;
    else if (d == "u8") o.type = ScalarType::Byte;
    else if (d == "i8") o.type = ScalarType::Char;
    else if (d == "bool") o.type = ScalarType::Bool;
    else return false;
    std::stringstream ss(s.substr(c2 + 1));
    for (std::string x; std::getline(ss, x, ',');) o.shape.push_back(std::atoi(x.c_str()));
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", file.c_str());
        return false;
    }
    o.data.assign(std::istreambuf_iterator<char>(in), {});
    return true;
}

std::vector<float> to_float(const executorch::aten::Tensor& t) {
    size_t n = t.numel();
    std::vector<float> f(n);
    auto conv = [&](auto tag) {
        using T = decltype(tag);
        const T* p = t.const_data_ptr<T>();
        for (size_t i = 0; i < n; ++i) f[i] = static_cast<float>(p[i]);
    };
    switch (t.scalar_type()) {
        case ScalarType::Float: conv(float{}); break;
        case ScalarType::Int: conv(int32_t{}); break;
        case ScalarType::Long: conv(int64_t{}); break;
        case ScalarType::Byte: conv(uint8_t{}); break;
        case ScalarType::Char: conv(int8_t{}); break;
        case ScalarType::Bool: conv(bool{}); break;
        default: std::fprintf(stderr, "output dtype %d not converted\n", static_cast<int>(t.scalar_type()));
    }
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: et-run <model.pte> <method> -- <out_prefix> <file>:<dtype>:<d0,...> ... [-- ...]\n"
                             "       et-run <model.pte> <method> <input.f32> <d0,d1,...> [output.f32]\n");
        return 2;
    }
    std::vector<std::pair<std::string, std::vector<Input>>> runs;
    if (std::string(argv[3]) != "--") {
        Input in;
        if (!parse(std::string(argv[3]) + ":f32:" + argv[4], in)) return 2;
        runs.push_back({argc > 5 ? argv[5] : "", {}});
        runs.back().second.push_back(std::move(in));
    } else {
        for (int a = 3; a < argc; ++a) {
            if (std::string(argv[a]) == "--") {
                if (a + 1 >= argc) break;
                runs.push_back({argv[++a], {}});
                continue;
            }
            Input in;
            if (runs.empty() || !parse(argv[a], in)) {
                std::fprintf(stderr, "bad argument: %s\n", argv[a]);
                return 2;
            }
            runs.back().second.push_back(std::move(in));
        }
    }
    const bool legacy = std::string(argv[3]) != "--";
    Module m(argv[1], Module::LoadMode::Mmap);
    if (auto e = m.load_method(argv[2]); e != executorch::runtime::Error::Ok) {
        std::fprintf(stderr, "load_method(%s) failed: 0x%x\n", argv[2], static_cast<unsigned>(e));
        return 1;
    }
    for (auto& [prefix, inputs] : runs) {
        std::vector<executorch::extension::TensorPtr> keep;
        std::vector<EValue> args;
        for (auto& in : inputs) {
            keep.push_back(executorch::extension::from_blob(in.data.data(), in.shape, in.type));
            args.emplace_back(*keep.back());
        }
        auto r = m.execute(argv[2], args);
        if (!r.ok()) {
            std::fprintf(stderr, "execute(%s) failed: 0x%x\n", argv[2], static_cast<unsigned>(r.error()));
            return 1;
        }
        for (size_t i = 0; i < r->size(); ++i) {
            if (!r->at(i).isTensor()) continue;
            auto f = to_float(r->at(i).toTensor());
            size_t best = 0;
            for (size_t j = 1; j < f.size(); ++j)
                if (f[j] > f[best]) best = j;
            if (legacy) {
                std::printf("outputs=%zu argmax=%zu score=%.6f\n", f.size(), best, f.empty() ? 0.0f : f[best]);
                if (!prefix.empty())
                    std::ofstream(prefix, std::ios::binary)
                        .write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
                break;
            }
            std::printf("%s out[%zu] n=%zu argmax=%zu max=%.6g\n", prefix.c_str(), i, f.size(), best,
                        f.empty() ? 0.0 : f[best]);
            std::ofstream(prefix + "." + std::to_string(i) + ".f32", std::ios::binary)
                .write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
        }
    }
    return 0;
}
