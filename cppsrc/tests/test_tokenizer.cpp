// ClipTokenizer が、OpenVINO の pe_tokenizer.xml（Python 版が使うもの）と同じ ID を返すことの確認。
// 正解データは scripts/gen_tokenizer_cases.py で作る（tests/data/tokenizer_cases.json）。
// 使い方: test_tokenizer [merges.txt] [tokenizer_cases.json]
#include <cstdio>
#include <fstream>
#include <sstream>

#include "common.h"
#include "tokenizer.h"
#include "yaml_lite.h"

#ifndef PES_SOURCE_DIR
#define PES_SOURCE_DIR "."
#endif

static std::string read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("開けません: %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    const std::string merges_path = argc > 1 ? argv[1] : std::string(PES_SOURCE_DIR) + "/data/clip_bpe_merges.txt";
    const std::string cases_path = argc > 2 ? argv[2] : std::string(PES_SOURCE_DIR) + "/tests/data/tokenizer_cases.json";
    try {
        pes::ClipTokenizer tokenizer(read_all(merges_path));
        const pes::yaml::Node root = pes::yaml::parse_file(cases_path);
        const pes::yaml::Node* cases = root.get("cases");
        if (!cases) {
            std::printf("cases がありません\n");
            return 2;
        }
        int failed = 0;
        for (const pes::yaml::Node& c : cases->list) {
            const std::string& text = c.get("text") ? c.get("text")->scalar : std::string();
            const pes::yaml::Node* ids = c.get("ids");
            std::vector<int64_t> expected;
            for (const pes::yaml::Node& v : ids->list) {
                int x;
                v.to_int(x);
                expected.push_back(x);
            }
            std::vector<int64_t> got(expected.size());
            tokenizer.encode(text, static_cast<int>(expected.size()), got.data());
            if (got != expected) {
                if (++failed <= 10) {
                    std::printf("不一致: \"%s\"\n  期待:", text.c_str());
                    for (auto v : expected) std::printf(" %lld", (long long)v);
                    std::printf("\n  実際:");
                    for (auto v : got) std::printf(" %lld", (long long)v);
                    std::printf("\n");
                }
            }
        }
        std::printf("test_tokenizer: %zu 件中 %d 件が不一致\n", cases->list.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("エラー: %s\n", e.what());
        return 2;
    }
}
