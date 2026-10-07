#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

class CountingSemaphore {
public:
    explicit CountingSemaphore(int slots) : free_(slots) {}
    bool acquire_for(std::chrono::milliseconds wait) {
        std::unique_lock<std::mutex> lock(m_);
        if (!cv_.wait_for(lock, wait, [&] { return free_ > 0; })) return false;
        --free_;
        return true;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(m_);
            ++free_;
        }
        cv_.notify_one();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    int free_;
};

// Releases a slot that was already acquired.
class SlotGuard {
public:
    explicit SlotGuard(CountingSemaphore& s) : s_(s) {}
    ~SlotGuard() { s_.release(); }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;

private:
    CountingSemaphore& s_;
};
