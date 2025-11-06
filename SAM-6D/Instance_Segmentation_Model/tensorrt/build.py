## Install
# uv pip install pycuda
# uv pip install tensorrt-cu12
# uv pip install onnx onnxruntime

import os
from segment_anything.modeling import Sam
from segment_anything import (
    sam_model_registry
)
import torch
import argparse
import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from SAM model."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="Type of SAM model to build.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/segment-anything/",
        help="Path to the SAM model checkpoint.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision for TensorRT engine.",
    )

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model_type2checkpoint_name = {"vit_l": "sam_vit_l_0b3195.pth", "vit_b": "sam_vit_b_01ec64.pth", "vit_h": "sam_vit_h_4b8939.pth"}
    checkpoint_path = os.path.join(args.checkpoint, model_type2checkpoint_name[args.model_type])
    sam = sam_model_registry[args.model_type](
        checkpoint=checkpoint_path
    )

    # Add TensorRT building logic here
    print(f"Successfully loaded SAM model of type {args.model_type} from {args.checkpoint}")

    model = sam.image_encoder
    model.eval()
    model = model.to(DEVICE)
    dummy_input = torch.randn(1, 3, 1024, 1024)
    dummy_input = dummy_input.to(DEVICE)
    
    # export to ONNX
    input_names=["input"]
    output_names=["image_embedding"]
    onnx_path = f"{args.checkpoint}/sam_{args.model_type}_embedding.onnx"
    if not os.path.exists(onnx_path):
        torch.onnx.export(model, 
                        dummy_input,
                        onnx_path,
                        input_names=input_names, 
                        output_names=output_names, 
                        dynamic_axes={"input": {0: "batch"}, "image_embeddings": {0: "batch"}},
                        opset_version=17)


    # export to TensorRT
    logger = trt.Logger(trt.Logger.WARNING)

    # Create builder, network, and parser
    builder = trt.Builder(logger)   
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    model_path = f"{args.checkpoint}/sam_{args.model_type}_embedding.onnx"
    success = parser.parse_from_file(model_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise RuntimeError("Failed to parse ONNX file.")
    
    import ipdb; ipdb.set_trace()
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,  4 * (1 << 30)) # 4 GiB

    # Enable FP16 if supported
    if builder.platform_has_fast_fp16 and args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)   
        print("FP16 mode enabled for TensorRT engine.")

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    engine_path = f"{args.checkpoint}/sam_{args.model_type}_embedding.trt"
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}")

