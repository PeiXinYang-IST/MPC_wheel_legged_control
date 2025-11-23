#pragma once

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() { reset(); }

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed(std::string A) const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - start_;
        std::cout << A << " cost " << diff.count() << "s" << std::endl;
        return diff.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};