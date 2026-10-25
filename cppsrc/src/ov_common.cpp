#include "ov_common.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "common.h"

namespace pes {

std::string upper_device_name(const std::string& device) {
    std::string out = device;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::toupper(c); });
    return out;
}

ov::AnyMap make_compile_config(int num_threads) {
    return {
        ov::inference_num_threads(num_threads),
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::enable_cpu_pinning(true),
        ov::num_streams(1),
    };
}

std::string model_path(const std::string& dir, const std::string& name) {
    std::string path = dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += name;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error(PES_ERROR_LOAD, "モデルファイルを開けません: " + path);
    return path;
}

std::string shape_to_string(const ov::Shape& shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) os << (i ? "," : "") << shape[i];
    os << "]";
    return os.str();
}

}  // namespace pes
