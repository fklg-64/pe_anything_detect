# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 公式の sam2/build_sam.py から変えた点（UPSTREAM.md も参照）:
#   - Hydra の compose（GlobalHydra に依存する）をやめ、OmegaConf.load + instantiate でモデルを組む。
#     core/torch/sam2（EdgeTAM）が GlobalHydra を先に初期化するため、2 つ目のパッケージでは compose の検索パスが使えない。
#   - image_size を引数で上書きできる（公式は yaml の 1024。このリポジトリは CLIP と共通の 512 で動かす）。
#   - 画像用だけを残した（ビデオ予測器・Hugging Face からのダウンロードは除く）。

import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PACKAGE_DIR = Path(__file__).resolve().parent


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    image_size=None,
    apply_postprocessing=True,
    **kwargs,
):
    """config_file: パッケージ内の相対パス（例: "configs/sam2.1/sam2.1_hiera_t.yaml"）または絶対パス。"""
    path = Path(config_file)
    cfg = OmegaConf.load(path if path.is_absolute() else PACKAGE_DIR / path)
    if image_size is not None:
        cfg.model.image_size = image_size
    if apply_postprocessing:
        # dynamically fall back to multi-mask if the single mask is not stable（公式の build_sam2 と同じ）
        cfg.model.sam_mask_decoder_extra_args = {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
