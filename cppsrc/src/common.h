// 内部で共有する小さなヘルパー（エラー型・処理時間の計測・メモリ確保）
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pe_segmentation.h"

namespace pes {

// PesStatus 付きの例外。C API の境界で捕まえて戻り値に変換する
class Error : public std::runtime_error {
public:
    Error(PesStatus status, const std::string& message) : std::runtime_error(message), status_(status) {}
    PesStatus status() const { return status_; }

private:
    PesStatus status_;
};

// ミリ秒単位で処理時間を計測するコンテキスト（Python 版の Timer）。verbose のときは名前と時間を標準出力に出す
class Timer {
public:
    Timer(std::string name, bool verbose, double* out_ms = nullptr)
        : name_(std::move(name)), verbose_(verbose), out_ms_(out_ms), start_(std::chrono::steady_clock::now()) {}
    ~Timer() {
        const double ms = elapsed_ms();
        if (out_ms_) *out_ms_ = ms;
        if (verbose_) std::printf("%-24s : %6.2f ms\n", name_.c_str(), ms);
    }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    std::string name_;
    bool verbose_;
    double* out_ms_;
    std::chrono::steady_clock::time_point start_;
};

// 64 バイト境界に揃えた float 配列（AVX2 のロード・OpenVINO の入力テンソル用）
struct AlignedFree {
    void operator()(void* p) const { std::free(p); }
};
template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedFree>;

template <typename T>
AlignedArray<T> make_aligned(size_t count) {
    size_t bytes = ((count * sizeof(T) + 63) / 64) * 64;
    if (bytes == 0) bytes = 64;
    void* p = std::aligned_alloc(64, bytes);
    if (!p) throw std::bad_alloc();
    return AlignedArray<T>(static_cast<T*>(p));
}

}  // namespace pes
