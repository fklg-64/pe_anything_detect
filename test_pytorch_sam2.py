from core.torch.feature_extractor import TorchCLIPBackend
from core.torch.sam2_runner import TorchSam2Backend
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_dataset_paths

if __name__ == "__main__":
    # 画像リスト（configs/dataset.yaml）
    image_paths = load_dataset_paths()

    # 設定データ（SAM2.1 用）
    config = load_config("./configs/torch_config_sam2.yaml")
    
    # 特徴量抽出モデル
    extractor = TorchCLIPBackend(config)

    # プロンプトセグメンテーションモデル（公式 SAM2.1）
    sam2 = TorchSam2Backend(config)

    # セグメンテーション実行処理インスタンス
    runner = SegmentationRunner(extractor, sam2, config)

    # セグメンテーション実行
    for path in image_paths:
        runner.run(image_path=path)
