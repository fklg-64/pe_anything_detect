# はじめに

- [perception_models](https://github.com/facebookresearch/perception_models.git)を活用して、様々な検出タスクを"高速"に行うことを目標にしたリポジトリ

- OpenVINOやfaissなども活用して、CPUで高速に動作する基盤モデルを目指す

## 使い方
1. OpenVINOモデルを生成

    ```bash
    python convert_openvino.py
    ```

2. 検査実行

    ```bash
    python develop.py
    ```

## 予定、checkpoint

- [ ] 前景、背景のセグメンテーション処理を開発
- [ ] セマンティックセグメンテーションを追加
- [ ] インスタンスセグメンテーション処理を追加

## 補足
- 現在はセグメンテーションモデルを開発中
- Pytorch もしくは OpenVINOでモデルを動作
- Pytorchでの動作は、基本的に確認程度とするため、実行しても処理が最新とは限らない