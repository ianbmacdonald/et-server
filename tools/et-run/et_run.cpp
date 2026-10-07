// et-run: run one method of a .pte on a raw float32 input file and print the output's argmax.
// Links exactly what et-server links (same archives, same operator registration library), so it proves
// that a given ExecuTorch build (e.g. a selective one) can run a model et-server does not serve.
//
//   et-run <model.pte> <method> <input.f32> <d0,d1,...> [output.f32]
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using executorch::extension::Module;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: et-run <model.pte> <method> <input.f32> <d0,d1,...> [output.f32]\n");
        return 2;
    }
    std::ifstream in(argv[3], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> x(raw.size() / sizeof(float));
    std::copy(raw.begin(), raw.begin() + x.size() * sizeof(float), reinterpret_cast<char*>(x.data()));
    std::vector<executorch::aten::SizesType> shape;
    std::stringstream ss(argv[4]);
    for (std::string d; std::getline(ss, d, ',');) shape.push_back(std::atoi(d.c_str()));
    Module m(argv[1], Module::LoadMode::Mmap);
    if (auto e = m.load_method(argv[2]); e != executorch::runtime::Error::Ok) {
        std::fprintf(stderr, "load_method(%s) failed: 0x%x\n", argv[2], static_cast<unsigned>(e));
        return 1;
    }
    auto t = executorch::extension::from_blob(x.data(), shape);
    auto r = m.execute(argv[2], t);
    if (!r.ok()) {
        std::fprintf(stderr, "execute(%s) failed: 0x%x\n", argv[2], static_cast<unsigned>(r.error()));
        return 1;
    }
    const auto& out = r->at(0).toTensor();
    const float* p = out.const_data_ptr<float>();
    size_t n = out.numel(), best = 0;
    for (size_t i = 1; i < n; ++i)
        if (p[i] > p[best]) best = i;
    std::printf("outputs=%zu argmax=%zu score=%.6f\n", n, best, p[best]);
    if (argc > 5) std::ofstream(argv[5], std::ios::binary).write(reinterpret_cast<const char*>(p), n * sizeof(float));
    return 0;
}
