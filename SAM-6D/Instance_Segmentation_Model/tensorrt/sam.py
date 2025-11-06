import os

from matplotlib.style import context
from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)
from segment_anything.modeling import Sam
from segment_anything.utils.amg import MaskData, generate_crop_boxes, rle_to_mask
import logging
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
from typing import Any, Dict, List, Optional, Tuple
import cv2
import torch.nn.functional as F

from predictor import ModifiedSamPredictor

from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

import tensorrt as trt
from utils import preprocess_image
import pycuda.driver as cuda
import pycuda.autoinit


class ModifiedSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        sam: Sam,
        min_mask_region_area: int = 0,
        points_per_batch: int = 64,
        stability_score_thresh: float = 0.85,
        stability_score_offset: float = 1.0,
        crop_nms_thresh: float = 0.7,
        box_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        segmentor_width_size=None,
        pred_iou_thresh: float = 0.88,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 1,
        output_mode: str = "binary_mask",
        points_per_side: Optional[int] = 32,
        point_grids: Optional[List[np.ndarray]] = None,
        trt_model_path: Optional[str] = None,
    ):

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = ModifiedSamPredictor(sam)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        self.segmentor_width_size = segmentor_width_size
        logging.info(f"Init CustomSamAutomaticMaskGenerator done!")

        self.use_tensorrt = False if trt_model_path is None else True
        if trt_model_path is not None:
            # load tensorrt model for image encoder
            TRT_LOGGER = trt.Logger()
            runtime = trt.Runtime(TRT_LOGGER)

            if not os.path.isfile(trt_model_path):
                raise FileNotFoundError(f"Could not find model in path\n{trt_model_path}")
            with open(trt_model_path, "rb") as f:
                serialized_engine = f.read()

            engine = runtime.deserialize_cuda_engine(serialized_engine)
            self.context = engine.create_execution_context()

            
            tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            self.input_buffer = {}
            self.input_memory = {}
            self.output_buffer = {}
            self.output_memory = {}
            for tensor in tensor_names:
                size = trt.volume(self.context.get_tensor_shape(tensor))
                dtype = trt.nptype(engine.get_tensor_dtype(tensor))
                print(f"Tensor: {tensor}, Size: {size}, Dtype: {dtype}, Mode: {engine.get_tensor_mode(tensor)}")

                if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(tensor, (1, 3, 1024, 1024))
                    self.input_buffer[tensor] = cuda.pagelocked_empty(size, dtype)
                    self.input_memory[tensor] = cuda.mem_alloc(self.input_buffer[tensor].nbytes)
                    self.context.set_tensor_address(tensor, int(self.input_memory[tensor]))
                else: # OUTPUT
                    self.output_buffer[tensor] = cuda.pagelocked_empty(size, dtype)
                    self.output_memory[tensor] = cuda.mem_alloc(self.output_buffer[tensor].nbytes)
                    self.context.set_tensor_address(tensor, int(self.output_memory[tensor]))



    def preprocess_resize(self, image: np.ndarray):
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)  # (width, height)
        )
        return resized_image

    def postprocess_resize(self, detections, orig_size):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(
            detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
        )
        detections["boxes"][:, [1, 3]] = torch.clamp(
            detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
        )
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)
        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data["masks"] = [torch.from_numpy(rle_to_mask(rle)) for rle in data["rles"]]
        data["masks"] = torch.stack(data["masks"])
        return {"masks": data["masks"].to(data["boxes"].device), "boxes": data["boxes"]}

    def remove_small_detections(self, mask_data: MaskData, img_size: List) -> MaskData:
        # calculate area and number of pixels in each mask
        area = box_area(mask_data["boxes"]) / (img_size[0] * img_size[1])
        idx_selected = area >= self.mask_post_processing.min_box_size
        mask_data.filter(idx_selected)
        return mask_data
    
    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]

        if self.use_tensorrt:
            embeddings = self.get_image_embedding_tensorrt(cropped_im)
        else:
            embeddings = None

        self.predictor.set_image(cropped_im, embeddings=embeddings)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data
    
    def get_image_embedding_tensorrt(self, image: np.ndarray):
        pixel_mean = torch.tensor([123.675, 116.28, 103.53])
        pixel_std = torch.tensor([58.395, 57.12, 57.375])
        img_size = 1024
        input_for_tensorrt = preprocess_image(image, 1024, "cpu", pixel_mean, pixel_std, img_size)

        # run inference
        stream = cuda.Stream()
        # Copy input to pagelocked buffer
        np.copyto(self.input_buffer["input"], np.ascontiguousarray(input_for_tensorrt.cpu().numpy().astype(np.float32).ravel()))

        cuda.memcpy_htod_async(self.input_memory["input"], self.input_buffer["input"], stream)
        self.context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.output_buffer["image_embedding"], self.output_memory["image_embedding"], stream)
        stream.synchronize()
        output = self.output_buffer["image_embedding"].reshape((1, 256, 64, 64))

        return torch.tensor(output).to(self.predictor.device)