# 入力が画像サイズ
IMAGE_SIZE: int = 512
# Kmeansのクラスタ数
K_CLUSTERS: int = 12
# Kmeansのループ回数（多いほどクラスタは整う）
N_ITER: int = 10
# モデルのディレクトリ
MODEL_DIR: str = "./weights/openvino_models_8_9_10"
# プロンプトが保存されているディレクトリ
PROMPTS_DIR: str="./prompts"
# OpenVINOの処理に使用するスレッド数
OV_NUM_THREAD: int = 4
# INT8で量子化されたモデルを使用するか？
USE_INT8: bool = True