# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 公式の sam2/__init__.py は Hydra（initialize_config_module("sam2")）を初期化するが、ここでは初期化しない。
# GlobalHydra はプロセスで 1 つで、core/torch/sam2（EdgeTAM）が先に初期化するため、2 つ目のパッケージが
# 同じ方法で初期化しても設定の検索パスは増えない。モデルは build_sam.py が OmegaConf で組む。
