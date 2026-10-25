// 設定 yaml とプロンプト yaml を読むための最小の YAML パーサー。
// 対応するのは本リポジトリの yaml が使う範囲:
//   - ブロック形式の「キー: 値」（ネストなし）と、キーの下の「- 要素」の並び
//   - フロー形式 { "キー": [ "要素", ... ], ... } / [ ... ]（複数行・末尾のカンマ・# コメント可）
//   - スカラーは "..." / '...' / 引用なし（true/false/数値/文字列）
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace pes::yaml {

struct Node {
    enum class Type { Null, Scalar, List, Map };
    Type type = Type::Null;
    std::string scalar;   // Scalar のとき（引用符は外してある）
    bool quoted = false;  // 引用符付きだったか（"true" を真偽値として扱わないため）
    std::vector<Node> list;
    std::vector<std::pair<std::string, Node>> map;

    bool is_null() const { return type == Type::Null; }
    const Node* get(const std::string& key) const;  // Map の子。無ければ nullptr

    // スカラーの変換。変換できなければ false（out は変更しない）
    bool to_int(int& out) const;
    bool to_double(double& out) const;
    bool to_bool(bool& out) const;
};

// 文字列から読む。形式が不正なら pes::Error(PES_ERROR_LOAD) を投げる
Node parse(const std::string& text);
Node parse_file(const std::string& path);

}  // namespace pes::yaml
