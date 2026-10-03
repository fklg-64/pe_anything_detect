# 入力が画像サイズ
IMAGE_SIZE: int = 512
# Kmeansのクラスタ数
K_CLUSTERS: list[int] = [8]
# Kmeansのループ回数（多いほどクラスタは整う）
N_ITER: int = 10
# モデルのディレクトリ
MODEL_DIR: str = "./weights/openvino_models"
# プロンプトが保存されているディレクトリ
PROMPTS_DIR: str="./prompts"
# OpenVINOの処理に使用するスレッド数
OV_NUM_THREAD: int = 4
# INT8で量子化されたモデルを使用するか？
USE_INT8: bool = False
# 前景と判断するための自信度合い
FOREGROUND_THRESHOLD: float = 0.5
# 結果を出力するディレクトリ
OUTPUT_DIR: str = "./outputs/experiment_results"