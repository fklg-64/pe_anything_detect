from core.utils import *
from core.openvino.feature_extractor import OVCLIPFeatureExtractor
from core.openvino.segmentation_runner import SegmentationRunner

from config import *

if __name__ == "__main__":
    image_paths = [
        "/home/llm/datasets/mvtec_anomaly_detection/capsule/test/good/000.png",
        "/home/llm/datasets/mvtec_anomaly_detection/hazelnut/test/good/001.png",
        "/home/llm/datasets/mvtec_anomaly_detection/grid/test/bent/002.png",
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
        "/home/llm/datasets/02_object_detection/coin/001_crop.png", 
        "/home/llm/datasets/02_object_detection/clip/002_crop.png", 
    ]

    extractor = OVCLIPFeatureExtractor(
        model_dir = MODEL_DIR,
        target_size=IMAGE_SIZE,
        use_int8 = USE_INT8, 
        device = "CPU", 
        num_threads = OV_NUM_THREAD
    )

    runner = SegmentationRunner(
        extractor, 
        prompts_dir=PROMPTS_DIR, 
        k_clusters=K_CLUSTERS,
        niter=N_ITER
    )

    for path in image_paths:
        runner.run(
            image_path=path,
            output_dir="./outputs/experiment_results",
        )