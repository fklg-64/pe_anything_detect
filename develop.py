from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.edgetam_runner import OVEdgeTamRunner
from core.common.segmentation_runner import SegmentationRunner
from core.utils import load_config

if __name__ == "__main__":
    # 画像リスト
    image_paths: list[str] = [
        "/home/llm/datasets/mvtec_anomaly_detection/capsule/test/good/000.png",
        "/home/llm/datasets/mvtec_anomaly_detection/hazelnut/test/good/001.png",
        # "/home/llm/datasets/mvtec_anomaly_detection/grid/test/bent/002.png",
        "/home/llm/datasets/01_classification_sample/noodle/noodle (1).png",
        "/home/llm/datasets/01_classification_sample/noodle/noodle (2).png",
        "/home/llm/datasets/01_classification_sample/noodle/noodle (3).png",
        "/home/llm/datasets/01_classification_sample/circuitboart/12-ato_1.png",
        "/home/llm/datasets/01_classification_sample/pin/OK/lightgreen (1).jpg", 
        "/home/llm/datasets/01_classification_sample/coin/OK/H25 (1).jpg", 
        "/home/llm/datasets/01_classification_sample/cap/OK/Yes (1).jpg", 
        "/home/llm/datasets/01_classification_sample/cable/OK/TypeC (5).jpg", 
        "/home/llm/datasets/01_classification_sample/bredboard/OK/cable1 (1).jpg", 
        "/home/llm/datasets/01_classification_sample/natural/pexels-rohit-george-1141376880-32113857.jpg",
        "/home/llm/datasets/01_classification_sample/natural/011kumakichi0327_TP_V.jpg", 
        "/home/llm/datasets/02_object_detection/syringe/006_syringe.png",
        "/home/llm/datasets/02_object_detection/coin/001_crop.png", 
        "/home/llm/datasets/02_object_detection/clip/002_crop.png", 
        "/home/llm/datasets/02_object_detection/clip/003_crop.png", 
    ]

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