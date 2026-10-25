#include "tokenizer.h"

#include <algorithm>
#include <sstream>

#include "common.h"

namespace pes {
namespace {

void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// UTF-8 の 1 文字を読む。不正なバイトは 1 バイトを 1 文字として返す
unsigned next_code_point(const std::string& s, size_t& i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    int len = 1;
    unsigned cp = c;
    if (c >= 0xF0 && c < 0xF8) { len = 4; cp = c & 0x07; }
    else if (c >= 0xE0) { len = 3; cp = c & 0x0F; }
    else if (c >= 0xC0) { len = 2; cp = c & 0x1F; }
    if (c >= 0x80 && c < 0xC0) len = 1;
    if (len == 1 || i + len > s.size()) { ++i; return c; }
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = static_cast<unsigned char>(s[i + k]);
        if ((cc & 0xC0) != 0x80) { ++i; return c; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += len;
    return cp;
}

// 小文字化（ASCII・Latin-1・Latin Extended-A・ギリシャ・キリル。HF トークナイザの Lowercase の主な範囲）
unsigned to_lower(unsigned cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp < 0x80) return cp;
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;
    if ((cp >= 0x100 && cp <= 0x12F) || (cp >= 0x132 && cp <= 0x137)) return (cp % 2 == 0) ? cp + 1 : cp;  // 0x130 (İ) は参照実装でも変わらない
    if (cp >= 0x139 && cp <= 0x148) return (cp % 2 == 1) ? cp + 1 : cp;
    if (cp >= 0x14A && cp <= 0x177) return (cp % 2 == 0) ? cp + 1 : cp;
    if (cp == 0x178) return 0xFF;
    if (cp >= 0x179 && cp <= 0x17E) return (cp % 2 == 1) ? cp + 1 : cp;
    if ((cp >= 0x391 && cp <= 0x3A1) || (cp >= 0x3A3 && cp <= 0x3AB)) return cp + 32;
    if (cp >= 0x410 && cp <= 0x42F) return cp + 32;
    if (cp >= 0x400 && cp <= 0x40F) return cp + 80;
    return cp;
}

bool is_space(unsigned cp) {
    return (cp >= 9 && cp <= 13) || cp == 32 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_number(unsigned cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp < 0x80) return false;
    return cp == 0xB2 || cp == 0xB3 || cp == 0xB9 || (cp >= 0xBC && cp <= 0xBE) || (cp >= 0x660 && cp <= 0x669) ||
           (cp >= 0x6F0 && cp <= 0x6F9) || (cp >= 0x966 && cp <= 0x96F) || (cp >= 0x2070 && cp <= 0x2079) ||
           (cp >= 0x2080 && cp <= 0x2089) || (cp >= 0x2150 && cp <= 0x2189) || (cp >= 0x2460 && cp <= 0x249B) ||
           (cp >= 0xFF10 && cp <= 0xFF19);
}

bool is_letter(unsigned cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if (cp < 0x80) return false;
    if (is_space(cp) || is_number(cp)) return false;
    if (cp >= 0x300 && cp <= 0x36F) return false;  // 結合文字（\p{M}）は \p{L} ではない
    // 記号・句読点の主な範囲（それ以外の非 ASCII は文字として扱う）
    if (cp <= 0xBF) return cp == 0xAA || cp == 0xB5 || cp == 0xBA;
    if (cp == 0xD7 || cp == 0xF7) return false;
    if ((cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x20A0 && cp <= 0x20CF) || (cp >= 0x2100 && cp <= 0x214F && cp != 0x2102 && cp != 0x2107) ||
        (cp >= 0x2190 && cp <= 0x2BFF) || (cp >= 0x2E00 && cp <= 0x2E7F) || (cp >= 0x3000 && cp <= 0x3004) ||
        (cp >= 0x3008 && cp <= 0x3020) || cp == 0x3030 || (cp >= 0xFE10 && cp <= 0xFE6F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
        (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65) ||
        (cp >= 0x1F000 && cp <= 0x1FAFF) || (cp >= 0xE000 && cp <= 0xF8FF))
        return false;
    return true;
}

struct CodePoint {
    unsigned cp;        // 小文字化した code point
    size_t begin, end;  // 小文字化後の文字列でのバイト範囲
};

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string& merges_text) {
    // OpenAI CLIP の bytes_to_unicode: 表示可能なバイトはそのまま、それ以外は 256 以降の文字に割り当てる
    std::vector<int> order;  // 語彙 ID 順のバイト
    for (int b = 33; b <= 126; ++b) order.push_back(b);
    for (int b = 161; b <= 172; ++b) order.push_back(b);
    for (int b = 174; b <= 255; ++b) order.push_back(b);
    std::vector<unsigned> code_of_byte(256, 0);
    for (int b : order) code_of_byte[b] = static_cast<unsigned>(b);
    unsigned extra = 256;
    for (int b = 0; b < 256; ++b) {
        if (code_of_byte[b] == 0) {
            code_of_byte[b] = extra++;
            order.push_back(b);
        }
    }

    // 語彙: 基本 256 個 -> "</w>" 付き 256 個 -> merges の順
    std::vector<std::string> symbols;
    for (int b : order) {
        std::string s;
        append_utf8(s, code_of_byte[b]);
        symbols.push_back(s);
    }
    for (size_t i = 0; i < 256; ++i) symbols.push_back(symbols[i] + "</w>");
    for (size_t i = 0; i < 256; ++i) byte_base_id_[order[i]] = static_cast<int>(i);

    std::unordered_map<std::string, int> ids;
    for (size_t i = 0; i < symbols.size(); ++i) ids.emplace(symbols[i], static_cast<int>(i));

    std::istringstream in(merges_text);
    std::string line;
    int rank = 0;
    while (std::getline(in, line) && rank < kNumMerges) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version", 0) == 0) continue;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) throw Error(PES_ERROR_LOAD, "BPE の merges の形式が不正です: " + line);
        const auto a = ids.find(line.substr(0, sp)), b = ids.find(line.substr(sp + 1));
        if (a == ids.end() || b == ids.end()) throw Error(PES_ERROR_LOAD, "BPE の merges に未知のトークンがあります: " + line);
        const std::string merged = line.substr(0, sp) + line.substr(sp + 1);
        int merged_id;
        const auto existing = ids.find(merged);
        if (existing != ids.end()) merged_id = existing->second;
        else {
            merged_id = static_cast<int>(symbols.size());
            symbols.push_back(merged);
            ids.emplace(merged, merged_id);
        }
        merges_.emplace((static_cast<uint64_t>(a->second) << 32) | static_cast<uint32_t>(b->second), Merge{rank, merged_id});
        ++rank;
    }
    if (rank != kNumMerges) throw Error(PES_ERROR_LOAD, "BPE の merges が足りません（" + std::to_string(rank) + " / " + std::to_string(kNumMerges) + " 行）");
}

std::vector<int> ClipTokenizer::bpe(const std::string& piece) const {
    const auto hit = cache_.find(piece);
    if (hit != cache_.end()) return hit->second;

    std::vector<int> word;
    for (size_t i = 0; i < piece.size(); ++i) {
        const int base = byte_base_id_[static_cast<unsigned char>(piece[i])];
        word.push_back(i + 1 == piece.size() ? base + 256 : base);  // 最後の文字は "</w>" 付き
    }
    while (word.size() > 1) {
        // 隣り合う組のうち、最も優先順位の高い（rank が小さい）もの
        int best_rank = -1;
        uint64_t best_key = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            const uint64_t key = (static_cast<uint64_t>(word[i]) << 32) | static_cast<uint32_t>(word[i + 1]);
            const auto m = merges_.find(key);
            if (m != merges_.end() && (best_rank < 0 || m->second.rank < best_rank)) {
                best_rank = m->second.rank;
                best_key = key;
            }
        }
        if (best_rank < 0) break;
        const int first = static_cast<int>(best_key >> 32), second = static_cast<int>(best_key & 0xFFFFFFFFu);
        const int merged = merges_.at(best_key).merged;
        std::vector<int> next;
        for (size_t i = 0; i < word.size();) {
            if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) { next.push_back(merged); i += 2; }
            else next.push_back(word[i++]);
        }
        word.swap(next);
    }
    cache_.emplace(piece, word);
    return word;
}

std::vector<int> ClipTokenizer::tokenize(const std::string& text) const {
    // 小文字化しながら code point に分ける（元のバイト列は BPE にそのまま渡す。小文字化で長さが変わる文字は無い範囲のみ扱う）
    std::string lowered;
    std::vector<CodePoint> cps;
    for (size_t i = 0; i < text.size();) {
        const size_t begin = i;
        const unsigned original = next_code_point(text, i);
        const unsigned cp = to_lower(original);
        CodePoint c{cp, lowered.size(), 0};
        // 不正な UTF-8 のバイト（code point = バイト値になっている）は、元のバイトをそのまま使う
        if (i - begin == 1 && static_cast<unsigned char>(text[begin]) >= 0x80) lowered += text[begin];
        else append_utf8(lowered, cp);
        c.end = lowered.size();
        cps.push_back(c);
    }

    // パターン  's|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+  の各選択肢を、先頭から順に試す
    static const char* const kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<int> ids;
    size_t n = cps.size();
    for (size_t i = 0; i < n;) {
        const unsigned cp = cps[i].cp;
        if (is_space(cp)) { ++i; continue; }
        size_t j = i;
        bool matched = false;
        if (cp == '\'') {
            for (const char* c : kContractions) {
                const size_t len = std::char_traits<char>::length(c);
                if (i + len <= n) {
                    bool ok = true;
                    for (size_t k = 0; k < len; ++k) ok = ok && cps[i + k].cp == static_cast<unsigned char>(c[k]);
                    if (ok) { j = i + len; matched = true; break; }
                }
            }
        }
        if (!matched) {
            if (is_letter(cp)) {
                j = i + 1;
                while (j < n && is_letter(cps[j].cp)) ++j;
            } else if (is_number(cp)) {
                j = i + 1;
            } else {
                j = i + 1;
                while (j < n && !is_space(cps[j].cp) && !is_letter(cps[j].cp) && !is_number(cps[j].cp)) ++j;
            }
        }
        const std::string piece = lowered.substr(cps[i].begin, cps[j - 1].end - cps[i].begin);
        const std::vector<int> pieces = bpe(piece);
        ids.insert(ids.end(), pieces.begin(), pieces.end());
        i = j;
    }
    return ids;
}

void ClipTokenizer::encode(const std::string& text, int context_length, int64_t* out) const {
    std::vector<int> body = tokenize(text);
    const int max_body = context_length - 2;
    if (static_cast<int>(body.size()) > max_body) body.resize(max_body);
    int pos = 0;
    out[pos++] = kStartOfText;
    for (int id : body) out[pos++] = id;
    out[pos++] = kEndOfText;
    while (pos < context_length) out[pos++] = 0;
}

}  // namespace pes
