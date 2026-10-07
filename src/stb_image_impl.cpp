// stb_image, compiled once, with a per-thread allocation budget.
//
// stb does not bound its own memory use by the image header: a PNG's zlib
// stream is inflated into a buffer that doubles until the stream ends
// (stbi__zexpand, up to UINT_MAX/2), whatever IHDR says, and a progressive JPEG
// keeps a coefficient plane per component. Routing STBI_MALLOC/REALLOC/FREE
// through a budget sized from the header turns those into a clean decode
// failure instead of an OOM kill.

#include "image_preprocess.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

struct Budget {
    bool active = false;
    bool exceeded = false;
    uint64_t limit = 0;
    uint64_t used = 0;
    uint64_t peak = 0;
};

thread_local Budget t_budget;

// 16 bytes keeps the returned pointer at malloc's own alignment.
constexpr size_t kHeader = 16;

size_t block_size(void* user) {
    size_t n;
    std::memcpy(&n, static_cast<unsigned char*>(user) - kHeader, sizeof(n));
    return n;
}

bool admit(uint64_t new_used) {
    Budget& b = t_budget;
    if (!b.active) return true;
    if (new_used > b.limit) {
        b.exceeded = true;
        return false;
    }
    return true;
}

void account(uint64_t new_used) {
    Budget& b = t_budget;
    if (!b.active) return;
    b.used = new_used;
    b.peak = std::max(b.peak, b.used);
}

}  // namespace

extern "C" {

static void* budget_malloc(size_t n) {
    if (n > SIZE_MAX - kHeader) return nullptr;
    const uint64_t new_used = t_budget.used + n;
    if (!admit(new_used)) return nullptr;
    auto* base = static_cast<unsigned char*>(std::malloc(n + kHeader));
    if (!base) return nullptr;
    std::memcpy(base, &n, sizeof(n));
    account(new_used);
    return base + kHeader;
}

static void budget_free(void* p) {
    if (!p) return;
    const size_t n = block_size(p);
    if (t_budget.active) t_budget.used -= std::min<uint64_t>(t_budget.used, n);
    std::free(static_cast<unsigned char*>(p) - kHeader);
}

// Charged as old + new: realloc may copy (glibc's dynamic mmap threshold puts
// later large blocks on the brk heap), so both blocks can be live at once.
static void* budget_realloc(void* p, size_t n) {
    if (!p) return budget_malloc(n);
    if (n > SIZE_MAX - kHeader) return nullptr;
    const size_t old = block_size(p);
    const uint64_t transient = t_budget.used + n;
    if (!admit(transient)) return nullptr;
    auto* base = static_cast<unsigned char*>(
        std::realloc(static_cast<unsigned char*>(p) - kHeader, n + kHeader));
    if (!base) return nullptr;
    std::memcpy(base, &n, sizeof(n));
    account(transient);
    if (t_budget.active) t_budget.used = transient - std::min<uint64_t>(transient, old);
    return base + kHeader;
}

}  // extern "C"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 16384
#define STBI_MALLOC(sz) budget_malloc(sz)
#define STBI_REALLOC(p, newsz) budget_realloc(p, newsz)
#define STBI_FREE(p) budget_free(p)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
// Two decodes can fail at once (--max-concurrent-decodes 2); stbi_failure_reason() is
// only per-request if stb keeps it per thread.
#ifndef STBI_THREAD_LOCAL
#error "stb_image must be built with thread-local failure reasons (do not define STBI_NO_THREAD_LOCALS)"
#endif

namespace stb_budget {

DecodeBudgetScope::DecodeBudgetScope(uint64_t limit) {
    t_budget = Budget{};
    t_budget.active = true;
    t_budget.limit = limit;
}

DecodeBudgetScope::~DecodeBudgetScope() { t_budget.active = false; }

uint64_t DecodeBudgetScope::peak() const { return t_budget.peak; }

bool DecodeBudgetScope::exceeded() const { return t_budget.exceeded; }

}  // namespace stb_budget
