#include "flat_json.h"

#include <cstdint>
#include <utility>

namespace {

using json = nlohmann::json;

constexpr std::size_t kMaxKeys = 32;

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

bool valid_utf8(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            continue;
        }
        size_t len;
        uint32_t cp;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > s.size()) return false;
        for (size_t k = 1; k < len; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
            cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += len;
    }
    return true;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool hex4(std::string_view s, size_t i, uint32_t& v) {
    if (i + 4 > s.size()) return false;
    v = 0;
    for (size_t k = 0; k < 4; ++k) {
        const char c = s[i + k];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
        else return false;
    }
    return true;
}

// Hand-rolled rather than nlohmann's lexer: the lexer grows its token buffer
// by doubling, which costs about 3x a large string's size at the peak. Here
// the output is reserved once from the raw span.
class Parser {
public:
    Parser(std::string_view s, std::string& error) : s_(s), error_(error) {}

    bool parse(json& out) {
        skip_ws();
        if (i_ == s_.size()) return fail("request body is not valid JSON");
        if (s_[i_] != '{') {
            const char c = s_[i_];
            const bool json_value = c == '[' || c == '"' || c == '-' || (c >= '0' && c <= '9') ||
                                    s_.substr(i_, 4) == "true" || s_.substr(i_, 5) == "false" ||
                                    s_.substr(i_, 4) == "null";
            return fail(json_value ? "request body must be a JSON object" : "request body is not valid JSON");
        }
        ++i_;
        skip_ws();
        if (peek('}')) {
            ++i_;
            return finish();
        }
        for (;;) {
            skip_ws();
            std::string key;
            if (!peek('"') || !string(key)) return fail("request body is not valid JSON");
            skip_ws();
            if (!peek(':')) return fail("request body is not valid JSON");
            ++i_;
            skip_ws();
            json v;
            if (!value(v)) return false;
            if (out.contains(key)) return fail("duplicate key '" + key.substr(0, 64) + "'");
            if (out.size() >= kMaxKeys) return fail("request body has more than 32 keys");
            out[key] = std::move(v);
            skip_ws();
            if (peek(',')) {
                ++i_;
                continue;
            }
            if (peek('}')) {
                ++i_;
                return finish();
            }
            return fail("request body is not valid JSON");
        }
    }

private:
    bool finish() {
        skip_ws();
        return i_ == s_.size() || fail("request body is not valid JSON");
    }

    bool value(json& v) {
        if (i_ == s_.size()) return fail("request body is not valid JSON");
        const char c = s_[i_];
        if (c == '{') return fail("request body must be a flat JSON object (no nested objects)");
        if (c == '[') return fail("request body must be a flat JSON object (no arrays)");
        if (c == '"') {
            std::string str;
            if (!string(str)) return fail("request body is not valid JSON");
            v = json(std::move(str));
            return true;
        }
        const size_t start = i_;
        while (i_ < s_.size() && s_[i_] != ',' && s_[i_] != '}' && !is_ws(s_[i_])) ++i_;
        const std::string_view token = s_.substr(start, i_ - start);
        if (token.empty() || token.size() > 64) return fail("request body is not valid JSON");
        v = json::parse(token.begin(), token.end(), nullptr, false);
        if (v.is_discarded() || v.is_structured() || v.is_string()) return fail("request body is not valid JSON");
        return true;
    }

    // s_[i_] is the opening quote.
    bool string(std::string& out) {
        size_t j = i_ + 1;
        bool escaped = false;
        while (j < s_.size() && s_[j] != '"') {
            if (static_cast<unsigned char>(s_[j]) < 0x20) return false;
            if (s_[j] == '\\') {
                escaped = true;
                ++j;
            }
            ++j;
        }
        if (j >= s_.size()) return false;
        const std::string_view raw = s_.substr(i_ + 1, j - i_ - 1);
        if (!valid_utf8(raw)) return false;
        i_ = j + 1;
        if (!escaped) {
            out.assign(raw.data(), raw.size());
            return true;
        }
        out.clear();
        out.reserve(raw.size());
        for (size_t k = 0; k < raw.size(); ++k) {
            if (raw[k] != '\\') {
                out.push_back(raw[k]);
                continue;
            }
            const char e = raw[++k];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(raw, k + 1, cp)) return false;
                    k += 4;
                    if (cp >= 0xDC00 && cp <= 0xDFFF) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        uint32_t lo;
                        if (k + 2 >= raw.size() || raw[k + 1] != '\\' || raw[k + 2] != 'u' || !hex4(raw, k + 3, lo) ||
                            lo < 0xDC00 || lo > 0xDFFF) {
                            return false;
                        }
                        k += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return true;
    }

    void skip_ws() {
        while (i_ < s_.size() && is_ws(s_[i_])) ++i_;
    }
    bool peek(char c) const { return i_ < s_.size() && s_[i_] == c; }
    bool fail(std::string message) {
        if (error_.empty()) error_ = std::move(message);
        return false;
    }

    std::string_view s_;
    std::string& error_;
    size_t i_ = 0;
};

}  // namespace

bool parse_flat_json_object(std::string_view body, json& out, std::string& error) {
    out = json::object();
    error.clear();
    Parser p(body, error);
    if (p.parse(out)) return true;
    if (error.empty()) error = "request body is not valid JSON";
    out = json::object();
    return false;
}
