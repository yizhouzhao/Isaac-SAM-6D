import os
import numpy as np
import imageio
import logging
import torch
from PIL import Image
from segment_anything import sam_model_registry
# from Instance_Segmentation_Model.model.sam import CustomSamAutomaticMaskGenerator
from sam import ModifiedSamAutomaticMaskGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_weight_dict = {
    "vit_l": "sam_vit_l_0b3195.pth",  # 1250MB
    "vit_b": "sam_vit_b_01ec64.pth",  # 375MB
    "vit_h": "sam_vit_h_4b8939.pth",  # 2500MB
}

def load_detector():
    model_type="vit_h"
    checkpoint_dir="./checkpoints/segment-anything/"

    sam = sam_model_registry[model_type](
        checkpoint=os.path.join(checkpoint_dir, pretrained_weight_dict[model_type])
    ).to(DEVICE)

    segmentor_model = ModifiedSamAutomaticMaskGenerator(
        sam=sam,
        min_mask_region_area=0,
        points_per_batch=64,
        stability_score_thresh=0.85,
        box_nms_thresh=0.7,
        segmentor_width_size=640,
        pred_iou_thresh=0.88,
    )

    return segmentor_model

    # descriptor_model = CustomDINOv2(
    #     model_name = "dinov2_vitl14",
    #     token_name = "x_norm_clstoken",
    #     descriptor_width_size=640,
    #     checkpoint_dir="./Instance_Segmentation_Model/checkpoints/dinov2/",
    #     image_size=224,
    #     chunk_size=16,
    #     validpatch_thresh=0.5,
    # )

    # detector = Detector(
    #     segmentor_model=segmentor_model,
    #     descriptor_model=descriptor_model,
    #     onboarding_config=onboarding_config,
    #     matching_config=matching_config,
    #     post_processing_config=post_processing_config,
    #     log_interval=5,
    #     log_dir="./logs/sam",
    #     visible_thred=0.5,
    #     pointcloud_sample_num=2048,
    # )

    
    # detector.descriptor_model.model = detector.descriptor_model.model.to(device)
    # detector.descriptor_model.model.device = device
    # # if there is predictor in the model, move it to device
    # if hasattr(detector.segmentor_model, "predictor"):
    #     detector.segmentor_model.predictor.model = (
    #         detector.segmentor_model.predictor.model.to(device)
    #     )
    # else:
    #     detector.segmentor_model.model.setup_model(device=device, verbose=True)
    # logging.info(f"Loading detector model to {device} done!")

    # return detector


if __name__ == "__main__":
    # Test
    rgb_path: str = "../Data/Example6/isaacsim_camera_capture_19_left.png"
    depth_path: str = "../Data/Example6/depth_map.png"
    det_score_thresh: float = 0.5

    rgb = Image.open(rgb_path).convert("RGB")
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)

    segmentor_model = load_detector()
    print("Loaded segmentor model:", segmentor_model)

    detections = segmentor_model.generate_masks(np.array(rgb))
    print("Generated masks:", len(detections))

    import ipdb; ipdb.set_trace()