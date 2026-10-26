from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config, load_dataset_paths

if __name__ == "__main__":
    # 画像リスト（configs/dataset.yaml）
    image_paths = load_dataset_paths()

    # 設定データ
    config = load_config()

    # 特徴量抽出モデル
    extractor = OVCLIPFeatureExtractor(config)

    # プロンプトセグメンテーションモデル
    edgetam = OVEdgeTamRunner(config)

    # セグメンテーション実行処理インスタンス
    runner = SegmentationRunner(extractor, edgetam, config)

    # セグメンテーション実行
    for path in image_paths:
        runner.run(image_path=path)