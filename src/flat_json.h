#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

// Parses a request body that must be one JSON object of scalar values
// ({"image": "...", "top_k": 3}). Nesting, a non-object top level, duplicate
// keys, more than 32 keys, invalid UTF-8 and malformed JSON fail with a
// message in `error`. Nothing but the scalars is materialized: a hostile body
// of nested arrays is refused at its first bracket, and a string value costs
// one allocation of its own size.
bool parse_flat_json_object(std::string_view body, nlohmann::json& out, std::string& error);
