"""docs/*.md の簡易検査: 画像・相対リンクの存在、Mermaid の構文（引用符・括弧・subgraph/loop と end の対応）、画像の合計サイズ。

    python tools/docs/lint_docs.py [--docs docs] [--max-total-mb 5]

Mermaid の描画はしない（mermaid-cli が無い環境向けの、典型的な誤りだけを拾う検査）。問題があれば終了コード 1。
"""
import argparse
import re
import sys
from pathlib import Path


def lint_mermaid(md: Path, text: str) -> int:
    problems = 0
    for match in re.finditer(r"```mermaid\n(.*?)```", text, re.S):
        block = match.group(1)
        first_line = text[:match.start()].count("\n") + 2
        kind = block.split("\n", 1)[0].split()[0]
        opens = ends = 0
        for offset, line in enumerate(block.split("\n")):
            where = f"{md}:{first_line + offset}"
            if line.count('"') % 2:
                print(f"{where}: 引用符が奇数: {line.strip()}")
                problems += 1
            outside = re.sub(r'"[^"]*"', "", line)  # 引用符の中は数えない
            if kind == "flowchart":
                for open_, close in ("[]", "{}", "()"):
                    if outside.count(open_) != outside.count(close):
                        print(f"{where}: 括弧 {open_}{close} が対応しない: {line.strip()}")
                        problems += 1
            token = line.strip()
            if token.startswith("subgraph") or (kind == "sequenceDiagram" and re.match(r"(loop|alt|opt|par)\b", token)):
                opens += 1
            if token == "end":
                ends += 1
        if opens != ends:
            print(f"{md}:{first_line}: subgraph/loop と end の数が違う（{opens} と {ends}）")
            problems += 1
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--max-total-mb", type=float, default=5.0, help="docs/images の合計サイズの上限（git に載せるため）")
    args = parser.parse_args()
    docs = Path(args.docs)

    problems = 0
    for md in sorted(docs.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if target.startswith(("http://", "https://")):
                continue
            if not (docs / target).exists():
                print(f"{md}: リンク先がない: {target}")
                problems += 1
        problems += lint_mermaid(md, text)

    images = [p for p in (docs / "images").glob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in images) / 1e6
    print(f"画像 {len(images)} 枚、合計 {total_mb:.2f} MB（上限 {args.max_total_mb} MB）")
    if total_mb > args.max_total_mb:
        print("画像が大きすぎる: python tools/docs/shrink_images.py（または make_doc_images.py で再生成）")
        problems += 1
    for png in (docs / "images").glob("*.png"):
        print(f"PNG が残っている: {png}（shrink_images.py で JPEG にする）")
        problems += 1

    print("問題なし" if not problems else f"問題 {problems} 件")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
