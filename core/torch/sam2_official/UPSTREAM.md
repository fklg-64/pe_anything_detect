# sam2_official

公式 SAM2.1（https://github.com/facebookresearch/sam2）のコピー。EdgeTAM の fork である `core/torch/sam2` とは別物。

- コピー元: `/home/llm/sam2/sam2`（git `2b90b9f` remove `.pin_memory()` in `obj_pos` of `SAM2Base` to resolve and error in MPS (#495)）
- 作り方: `python tools/setup/vendor_sam2.py`（再実行すると、下の「公式から変えていないファイル」を作り直す）
- 公式から変えたのは次だけ。
  - パッケージ名 `sam2` → `sam2_official`（Python の `from sam2.` / `import sam2` と、yaml の `_target_: sam2.`）。コードの中身は同じ。
  - `__init__.py`: Hydra（`initialize_config_module("sam2")`）を初期化しない。`GlobalHydra` はプロセスで 1 つで、`core/torch/sam2`（EdgeTAM）が先に初期化するため。
  - `build_sam.py`: Hydra の `compose` を使わず、`OmegaConf.load` + `instantiate` でモデルを組む。`image_size` を引数で上書きできる（公式は yaml の値）。
- コピーしていないもの: ビデオ予測器・自動マスク生成・ベンチマーク・`csrc`・SAM2（1.0）と学習用の設定・`utils/amg.py`。`utils/misc.py` の `get_connected_components`（`from sam2_official import _C`）は C 拡張が無いので使えない（このリポジトリは使わない）。

## 公式から変えていないファイル（22）
- `modeling/__init__.py`
- `modeling/backbones/__init__.py`
- `modeling/backbones/hieradet.py`
- `modeling/backbones/image_encoder.py`
- `modeling/backbones/utils.py`
- `modeling/memory_attention.py`
- `modeling/memory_encoder.py`
- `modeling/position_encoding.py`
- `modeling/sam/__init__.py`
- `modeling/sam/mask_decoder.py`
- `modeling/sam/prompt_encoder.py`
- `modeling/sam/transformer.py`
- `modeling/sam2_base.py`
- `modeling/sam2_utils.py`
- `utils/__init__.py`
- `utils/misc.py`
- `utils/transforms.py`
- `sam2_image_predictor.py`
- `configs/sam2.1/sam2.1_hiera_b+.yaml`
- `configs/sam2.1/sam2.1_hiera_l.yaml`
- `configs/sam2.1/sam2.1_hiera_s.yaml`
- `configs/sam2.1/sam2.1_hiera_t.yaml`
