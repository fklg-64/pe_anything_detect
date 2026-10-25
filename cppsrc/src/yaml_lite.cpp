#include "yaml_lite.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "common.h"

namespace pes::yaml {

const Node* Node::get(const std::string& key) const {
    for (const auto& kv : map)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

bool Node::to_int(int& out) const {
    if (type != Type::Scalar || quoted || scalar.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(scalar.c_str(), &end, 10);
    if (errno != 0 || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

bool Node::to_double(double& out) const {
    if (type != Type::Scalar || quoted || scalar.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(scalar.c_str(), &end);
    if (errno != 0 || *end != '\0') return false;
    out = v;
    return true;
}

bool Node::to_bool(bool& out) const {
    if (type != Type::Scalar || quoted) return false;
    std::string s = scalar;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "true" || s == "yes" || s == "on") { out = true; return true; }
    if (s == "false" || s == "no" || s == "off") { out = false; return true; }
    return false;
}

namespace {

[[noreturn]] void fail(const std::string& msg, size_t line) {
    throw Error(PES_ERROR_LOAD, "yaml の解析に失敗しました（" + std::to_string(line) + " 行付近）: " + msg);
}

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

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// フロー形式（{...} / [...] / スカラー）のパーサー。text 全体、または 1 つの値の文字列を読む
class FlowParser {
public:
    FlowParser(const std::string& text, size_t first_line = 1) : s_(text), line_(first_line) {}

    Node parse_value() {
        skip();
        if (eof()) return Node{};
        const char c = s_[pos_];
        if (c == '{') return parse_map();
        if (c == '[') return parse_list();
        return parse_scalar(false);
    }

    void skip() {  // 空白・改行・コメントを飛ばす
        while (!eof()) {
            const char c = s_[pos_];
            if (c == '\n') { ++line_; ++pos_; }
            else if (c == ' ' || c == '\t' || c == '\r') ++pos_;
            else if (c == '#' && (pos_ == 0 || std::isspace(static_cast<unsigned char>(s_[pos_ - 1])))) {
                while (!eof() && s_[pos_] != '\n') ++pos_;
            } else break;
        }
    }
    bool eof() const { return pos_ >= s_.size(); }

private:
    Node parse_map() {
        ++pos_;  // '{'
        Node node;
        node.type = Node::Type::Map;
        for (;;) {
            skip();
            if (eof()) fail("} が閉じていません", line_);
            if (s_[pos_] == '}') { ++pos_; break; }
            if (s_[pos_] == ',') { ++pos_; continue; }
            Node key = parse_scalar(true);
            skip();
            if (eof() || s_[pos_] != ':') fail("キーの後に : がありません", line_);
            ++pos_;
            node.map.emplace_back(key.scalar, parse_value());
        }
        return node;
    }

    Node parse_list() {
        ++pos_;  // '['
        Node node;
        node.type = Node::Type::List;
        for (;;) {
            skip();
            if (eof()) fail("] が閉じていません", line_);
            if (s_[pos_] == ']') { ++pos_; break; }
            if (s_[pos_] == ',') { ++pos_; continue; }
            node.list.push_back(parse_value());
        }
        return node;
    }

    // is_key のとき ':' で終わる。そうでなければ , ] } で終わる
    Node parse_scalar(bool is_key) {
        Node node;
        node.type = Node::Type::Scalar;
        const char c = s_[pos_];
        if (c == '"') {
            node.quoted = true;
            ++pos_;
            while (!eof() && s_[pos_] != '"') {
                char ch = s_[pos_++];
                if (ch == '\n') ++line_;
                if (ch != '\\') { node.scalar += ch; continue; }
                if (eof()) break;
                const char e = s_[pos_++];
                switch (e) {
                    case 'n': node.scalar += '\n'; break;
                    case 't': node.scalar += '\t'; break;
                    case 'r': node.scalar += '\r'; break;
                    case '0': node.scalar += '\0'; break;
                    case 'u': {
                        const unsigned cp = static_cast<unsigned>(std::strtoul(s_.substr(pos_, 4).c_str(), nullptr, 16));
                        append_utf8(node.scalar, cp);
                        pos_ += 4;
                        break;
                    }
                    default: node.scalar += e; break;  // \\ \" \/ など
                }
            }
            if (eof()) fail("\" が閉じていません", line_);
            ++pos_;
            return node;
        }
        if (c == '\'') {
            node.quoted = true;
            ++pos_;
            for (;;) {
                if (eof()) fail("' が閉じていません", line_);
                const char ch = s_[pos_++];
                if (ch == '\'') {
                    if (!eof() && s_[pos_] == '\'') { node.scalar += '\''; ++pos_; continue; }
                    break;
                }
                if (ch == '\n') ++line_;
                node.scalar += ch;
            }
            return node;
        }
        // 引用なし
        const size_t begin = pos_;
        while (!eof()) {
            const char ch = s_[pos_];
            if (ch == '\n' || ch == ',' || ch == ']' || ch == '}') break;
            if (is_key && ch == ':' && (pos_ + 1 >= s_.size() || std::isspace(static_cast<unsigned char>(s_[pos_ + 1])))) break;
            if (ch == '#' && pos_ > begin && std::isspace(static_cast<unsigned char>(s_[pos_ - 1]))) break;
            ++pos_;
        }
        node.scalar = trim(s_.substr(begin, pos_ - begin));
        if (node.scalar == "null" || node.scalar == "~" || node.scalar.empty()) node.type = Node::Type::Null;
        return node;
    }

    const std::string& s_;
    size_t pos_ = 0;
    size_t line_;
};

// 行末のコメント（引用符の外の、空白の直後の #）を取り除く
std::string strip_comment(const std::string& line) {
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quote) {
            if (c == '\\' && quote == '"') ++i;
            else if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(line[i - 1])))) {
            return line.substr(0, i);
        }
    }
    return line;
}

// [ ] { } の深さ（引用符の外）。複数行にまたがる値の終端を探すのに使う
int bracket_balance(const std::string& s) {
    int depth = 0;
    char quote = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote) {
            if (c == '\\' && quote == '"') ++i;
            else if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') quote = c;
        else if (c == '[' || c == '{') ++depth;
        else if (c == ']' || c == '}') --depth;
    }
    return depth;
}

Node parse_scalar_text(const std::string& text, size_t line) {
    FlowParser p(text, line);
    Node n = p.parse_value();
    return n;
}

Node parse_block(const std::string& text) {
    std::vector<std::string> lines;
    {
        std::istringstream in(text);
        std::string l;
        while (std::getline(in, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines.push_back(l);
        }
    }
    Node root;
    root.type = Node::Type::Map;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string raw = strip_comment(lines[i]);
        const std::string line = trim(raw);
        if (line.empty() || line == "---" || line == "...") continue;
        if (line.rfind("- ", 0) == 0 || line == "-") fail("キーの無い配列の要素です", i + 1);

        // キーを探す（引用符付きキーも許す）
        std::string key;
        size_t value_begin = std::string::npos;
        if (line[0] == '"' || line[0] == '\'') {
            const char q = line[0];
            size_t j = 1;
            while (j < line.size() && line[j] != q) ++j;
            if (j >= line.size()) fail("キーの引用符が閉じていません", i + 1);
            key = line.substr(1, j - 1);
            size_t colon = line.find(':', j);
            if (colon == std::string::npos) fail("「キー: 値」の形式ではありません", i + 1);
            value_begin = colon + 1;
        } else {
            size_t colon = std::string::npos;
            for (size_t j = 0; j < line.size(); ++j)
                if (line[j] == ':' && (j + 1 == line.size() || std::isspace(static_cast<unsigned char>(line[j + 1])))) { colon = j; break; }
            if (colon == std::string::npos) fail("「キー: 値」の形式ではありません: " + line, i + 1);
            key = trim(line.substr(0, colon));
            value_begin = colon + 1;
        }
        std::string value = trim(line.substr(value_begin));

        Node node;
        if (value.empty()) {
            // 次の行以降が「- 要素」ならブロック形式の配列
            size_t j = i + 1;
            Node list;
            list.type = Node::Type::List;
            for (; j < lines.size(); ++j) {
                const std::string item_line = trim(strip_comment(lines[j]));
                if (item_line.empty()) continue;
                if (item_line.rfind("- ", 0) != 0 && item_line != "-") break;
                list.list.push_back(parse_scalar_text(trim(item_line.substr(1)), j + 1));
            }
            if (!list.list.empty()) { node = std::move(list); i = j - 1; }
        } else {
            // 複数行にまたがるフロー形式（color_palette など）は、閉じるまで次の行を連結する
            std::string joined = value;
            size_t j = i;
            while (bracket_balance(joined) > 0 && j + 1 < lines.size()) joined += "\n" + strip_comment(lines[++j]);
            if (bracket_balance(joined) != 0) fail("括弧が閉じていません", i + 1);
            node = parse_scalar_text(joined, i + 1);
            i = j;
        }
        root.map.emplace_back(std::move(key), std::move(node));
    }
    return root;
}

}  // namespace

Node parse(const std::string& text) {
    // 先頭の空白・コメントを飛ばして { / [ で始まっていれば全体がフロー形式、そうでなければブロック形式のマッピング
    size_t first = 0;
    for (;;) {
        while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
        if (first < text.size() && text[first] == '#') {
            while (first < text.size() && text[first] != '\n') ++first;
            continue;
        }
        break;
    }
    if (first < text.size() && (text[first] == '{' || text[first] == '[')) {
        FlowParser parser(text);
        Node n = parser.parse_value();
        parser.skip();
        if (!parser.eof()) fail("末尾に解釈できない文字があります", 0);
        return n;
    }
    return parse_block(text);
}

Node parse_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error(PES_ERROR_LOAD, "yaml を開けません: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    try {
        return parse(ss.str());
    } catch (const Error& e) {
        throw Error(e.status(), path + ": " + e.what());
    }
}

}  // namespace pes::yaml
