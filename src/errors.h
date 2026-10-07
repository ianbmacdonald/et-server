#pragma once

#include <stdexcept>

// Exceptions the HTTP layer maps to status codes. Anything else is a 500.
struct InvalidInput : std::runtime_error {  // 400: the request is at fault
    using std::runtime_error::runtime_error;
};

struct PayloadTooLarge : std::runtime_error {  // 413
    using std::runtime_error::runtime_error;
};

struct Busy : std::runtime_error {  // 503: no decode slot within the wait limit
    using std::runtime_error::runtime_error;
};
