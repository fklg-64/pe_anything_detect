# はじめに

- [perception_models](https://github.com/facebookresearch/perception_models.git)を活用して、様々な検出タスクを"高速"に行うことを目標にしたリポジトリ

- OpenVINOやfaissなども活用して、CPUで高速に動作する基盤モデルを目指す

## 使い方

### OpenVINOモデル

1. PyTorch -> OpenVINOモデルに変換

    ```bash
    python convert_openvino.py
    ```

2. モデル実行

    ```bash
    python develop.py
    ```

### C++ 版（OpenVINO）

`cppsrc/` に C++ への書き起こし（共有ライブラリ + サンプル）がある。詳細は [cppsrc/README.md](cppsrc/README.md)。

```bash
cd cppsrc && ./build.sh && cd ..
./cppsrc/build/bin/pe_segmentation_sample image.png
```

### PyTorchモデル

1. モデル実行

    ```bash
    python test_pytorch.py
    ```

## 予定、checkpoint

- [x] 前景、背景のセグメンテーション処理を開発
- [x] セマンティックセグメンテーションを追加
- [ ] インスタンスセグメンテーション処理を追加
- [ ] 処理時間の高速化（～200ms）

## 補足
- PyTorchでの動作は、基本的に確認程度とするため、実行しても処理が最新とは限らない

### インスタンス分離に関して

同じクラスの領域を、今回のスコアマップや EdgeTAM のロジットから分けることになります。有力なのは次の 2 つです。

- 領域を分けてから EdgeTAM に渡す方法です。クラスごとのマスクで距離変換して極大点を取り、各点を正例にして EdgeTAM を個別に走らせ、overlap_prior_floor と同じ要領で重なりを解決します。
- 前景クラスマスクにそのまま watershed をかける方法です。軽くて速い反面、形が不規則だと過分割になります。

どちらも、くっついた同種の物体の境界は距離変換だけでは決まりにくいのが難点です。