from core.torch.feature_extractor import TorchCLIPBackend
from core.torch.edgetam_runner import TorchEdgeTamBackend
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_dataset_paths

if __name__ == "__main__":
    # 画像リスト（configs/dataset.yaml）
    image_paths = load_dataset_paths()

    # 設定データ
    config = load_config("./configs/torch_config.yaml")
    
    # 特徴量抽出モデル
    extractor = TorchCLIPBackend(config)

    # プロンプトセグメンテーションモデル
    edgetam = TorchEdgeTamBackend(config)

    # セグメンテーション実行処理インスタンス
    runner = SegmentationRunner(extractor, edgetam, config)

    # セグメンテーション実行
    for path in image_paths:
        runner.run(image_path=path)