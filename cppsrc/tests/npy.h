// テスト用の最小の .npy 読み書き（C 順・リトルエンディアンの f4 / i4 / i8 / u1 / b1 だけ）
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
    std::string dtype;  // 例: "<f4"
    std::vector<size_t> shape;
    std::vector<char> data;
    size_t size() const {
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n;
    }
    template <typename T>
    const T* ptr() const { return reinterpret_cast<const T*>(data.data()); }
};

inline Array load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("開けません: " + path);
    char magic[6];
    in.read(magic, 6);
    unsigned char ver[2];
    in.read(reinterpret_cast<char*>(ver), 2);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t l;
        in.read(reinterpret_cast<char*>(&l), 2);
        header_len = l;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, ' ');
    in.read(&header[0], header_len);
    Array a;
    const size_t d = header.find("'descr':");
    const size_t q1 = header.find('\'', d + 8), q2 = header.find('\'', q1 + 1);
    a.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    const size_t s = header.find("'shape':");
    const size_t p1 = header.find('(', s), p2 = header.find(')', p1);
    std::stringstream ss(header.substr(p1 + 1, p2 - p1 - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoul(tok));
    }
    if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("Fortran 順は未対応: " + path);
    size_t item = 1;
    if (a.dtype == "<f4" || a.dtype == "<i4") item = 4;
    else if (a.dtype == "<i8" || a.dtype == "<f8") item = 8;
    else if (a.dtype == "|u1" || a.dtype == "|b1" || a.dtype == "|i1") item = 1;
    else throw std::runtime_error("未対応の dtype: " + a.dtype);
    a.data.resize(a.size() * item);
    in.read(a.data.data(), a.data.size());
    return a;
}

}  // namespace npy
