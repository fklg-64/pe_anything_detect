// CLIP の BPE トークナイザ（OpenVINO の pe_tokenizer.xml と同じ出力を、拡張ライブラリ無しで得る）。
// 正規化（小文字化）-> 正規表現相当の分割 -> バイト単位 BPE。語彙は merges から OpenAI CLIP と同じ規則で作る。
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pes {

class ClipTokenizer {
public:
    static constexpr int kStartOfText = 49406;
    static constexpr int kEndOfText = 49407;
    static constexpr int kNumMerges = 49152 - 256 - 2;  // 語彙 49408 = 512 + merges + 2 つの特殊トークン

    // merges.txt（"a b" の並び。先頭の #version 行は無視）の内容から作る
    explicit ClipTokenizer(const std::string& merges_text);

    // 本文のトークン ID（SOT / EOT / 詰め物を含まない）
    std::vector<int> tokenize(const std::string& text) const;

    // [SOT, 本文（context_length - 2 個まで）, EOT, 0 詰め] を context_length 個書く（pe_tokenizer.xml と同じ）
    void encode(const std::string& text, int context_length, int64_t* out) const;

private:
    std::vector<int> bpe(const std::string& piece) const;

    int byte_base_id_[256];  // バイト -> 語彙 ID（"</w>" 無し）。"</w>" 付きは +256
    // (左, 右) の ID の組 -> {優先順位, 結合後の ID}
    struct Merge { int rank; int merged; };
    std::unordered_map<uint64_t, Merge> merges_;
    mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

}  // namespace pes
