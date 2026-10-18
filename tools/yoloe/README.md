# YOLOE（テキストプロンプトのゼロショット検出・セグメンテーション）

テキストで物体の名前を与えるだけで、検出とセグメンテーションができる [YOLOE](https://docs.ultralytics.com/models/yoloe/) を、
このリポジトリのパイプライン（CLIP + KMeans + EdgeTAM）と比べるためのツール。

## フォルダ構成

```
tools/yoloe/
├── app.py            Web GUI のサーバー（FastAPI）
├── web/index.html    Web GUI の画面（単一ファイル、ビルド不要）
├── yoloe.py          CLI（画像とプロンプトを指定して実行）
├── yoloe_core.py     共通処理（推論・結果の整形・プロンプトJSONの読み書き）
├── prompts/
│   └── develop_prompts.json   develop.py の画像ごとのプロンプト（GUI で変更すると書き換わる）
└── weights/          重み（git 管理外）
    ├── yoloe-v8s-seg.pt       YOLOE のチェックポイント（*.pt を置くと GUI の選択肢に出る）
    └── mobileclip_blt.ts      テキストエンコーダ（無ければ初回に自動ダウンロード。約570MB）
```

## 準備

```bash
pip install ultralytics fastapi uvicorn python-multipart
pip install "git+https://github.com/ultralytics/CLIP.git"   # ultralytics のテキストエンコーダが使う
```

`weights/` に `yoloe-v8s-seg.pt` を置く。`mobileclip_blt.ts` は、GitHub からの取得が遅いと 1 時間近くかかる。
途中で止まったら、`weights/` で次を実行すると続きから取得できる。

```bash
curl -L -C - --retry 30 --retry-all-errors -o mobileclip_blt.ts \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts
```

## Web GUI

```bash
python tools/yoloe/app.py                  # http://localhost:8000（ブラウザが自動で開く）
python tools/yoloe/app.py --port 8080 --no-browser
python tools/yoloe/app.py --host 0.0.0.0   # Docker・別のPCのブラウザから開く場合
```

WSL では、Windows 側のブラウザから `http://localhost:8000` で開ける。customtkinter と違い、画面はブラウザが描くので、WSL・Docker でもレイアウトが崩れない。

流れ（左の 1〜4 の順）:

1. **モデル**: 起動時に `weights/yoloe-v8s-seg.pt` を読み込む。別の `*.pt` に切り替えて「読み込み」もできる。
2. **画像**: クリックで選択 / ドラッグ＆ドロップ / パスを指定 / develop.py の画像を選択（←→ キーで前後）。
3. **プロンプト**: `prompts/develop_prompts.json` を読み込み、画像名のキー（なければ `default`）の内容を表示する。名前を足す・消すと、**JSON に自動で書き戻す**（保存先のキーは切り替え可能）。並び順がクラスの色になる。
4. **検査**: conf・NMS の IoU・推論サイズを決めて「検査を実行」（`Ctrl+Enter`）。「変更で自動実行」を入れると、プロンプトを変えるたびに実行する。

ビューアの操作:

| 操作 | 内容 |
|---|---|
| マウスホイール | カーソル位置を中心に拡大縮小 |
| ドラッグ | 移動 |
| ダブルクリック / `F` | 全体表示 |
| `1` | 等倍 |
| `C` 長押し | 結果を隠して元画像を見る |
| `M` `O` `B` `L` | マスク / 輪郭 / ボックス / ラベルの表示切り替え |
| 検出にマウスを重ねる / クリック | その検出を強調（一覧と連動）。一覧の行をクリックすると、その検出へ拡大 |
| 右の一覧の目のアイコン | クラスごとの表示 / 非表示 |

結果の保存:

- 「結果を保存」: `outputs/yoloe_gui/<画像名>/` に `overlay.png`、`semantic_map.png`（クラス ID）、`instance_map.png`（検出ごとの ID）、`detections.json` を保存する。
- 「画像を保存」: 表示中の重畳画像（マスク・輪郭・ボックス・ラベル）を、元の解像度の PNG でダウンロードする。

## CLI

```bash
python tools/yoloe/yoloe.py --source <画像> --names coin clip
python tools/yoloe/yoloe.py --develop --names-json tools/yoloe/prompts/develop_prompts.json --conf 0.05
```

出力は `outputs/yoloe/<画像名>/`（GUI の「結果を保存」と同じ4ファイル）。

## プロンプトの JSON

```json
{
  "default": ["object"],
  "001_crop": ["coin"],
  "noodle (1)": ["instant noodles", "cup", "lid"]
}
```

キーは画像のファイル名（拡張子なし）。無いときは `default` を使う。

## 補足

- 確信度の既定は GUI が 0.10、CLI が 0.25。YOLOE のテキストプロンプトは確信度が低めに出るので、検出がないときは 0.05 程度まで下げる。
- 画像のパス指定 API は、このPC上の画像ファイルを読む。`--host 0.0.0.0` で公開する場合は、信頼できるネットワークでのみ使うこと。
