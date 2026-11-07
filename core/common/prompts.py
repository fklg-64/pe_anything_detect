"""前景/背景のテキストプロンプトと、そのテキスト特徴量（PE のテキストエンコーダの出力）。

粗い段階（クラスタ・ブロックの判定）と、精緻化の後の最終の名前付けは、同じ語彙・同じ照合（CLIP のロジット = CLIP_LOGIT_SCALE × コサイン類似度）で
物体の名前を決める。プロンプトは「前景 -> 背景」の順に並べ、プロンプト番号はその通し番号。クラスの名前は、前景プロンプトの番号で持つ。
"""

from pathlib import Path
from typing import List

import numpy as np
import yaml

from core.common.backend import CLIPBackend
from core.common.params import CLIP_LOGIT_SCALE
from core.utils import normalize_rows


def load_prompts(path: Path) -> List[str]:
    """プロンプトの yaml から、base（あれば）と custom を連結して返す。"""
    with open(path, mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("base", []) + data["custom"]


class PromptBank:
    """`prompts_dir` の foreground.yaml / background.yaml のプロンプトと、そのテキスト特徴量。"""

    def __init__(self, extractor: CLIPBackend, prompts_dir: str):
        prompts_dir = Path(prompts_dir)
        self.foreground: List[str] = load_prompts(prompts_dir / "foreground.yaml")
        self.background: List[str] = load_prompts(prompts_dir / "background.yaml")
        self.num_foreground: int = len(self.foreground)

        foreground_features = extractor.extract_text_features(self.foreground)
        background_features = extractor.extract_text_features(self.background)
        self.text_features: np.ndarray = np.vstack([foreground_features, background_features])  # 前景 -> 背景の順
        self.foreground_text_features: np.ndarray = self.text_features[: self.num_foreground]

    @property
    def all(self) -> List[str]:
        """プロンプト番号の順（前景 -> 背景）の全プロンプト。"""
        return self.foreground + self.background

    def logits(self, token: np.ndarray) -> np.ndarray:
        """代表トークン 1 つと、全プロンプト（前景 -> 背景の順）との類似度 × CLIP_LOGIT_SCALE。(前景 + 背景のプロンプト数,)"""
        return (CLIP_LOGIT_SCALE * normalize_rows(token.reshape(1, -1)) @ self.text_features.T)[0]

    def foreground_logits(self, tokens: np.ndarray) -> np.ndarray:
        """代表トークン (N, C) と、前景プロンプトとの類似度 × CLIP_LOGIT_SCALE。(N, 前景プロンプト数)"""
        return CLIP_LOGIT_SCALE * normalize_rows(tokens) @ self.foreground_text_features.T
