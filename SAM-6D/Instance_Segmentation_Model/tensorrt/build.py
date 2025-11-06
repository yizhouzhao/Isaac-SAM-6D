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
import numpy as np
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

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
        default="./checkpoints/segment-anything",
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
                        # dynamic_axes={"input": {0: "batch"}, "image_embeddings": {0: "batch"}},
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
    
    # import ipdb; ipdb.set_trace()
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,  4 * (1 << 30)) # 4 GiB

    # Enable FP16 if supported
    if builder.platform_has_fast_fp16 and args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)   
        print("FP16 mode enabled for TensorRT engine.")

    ## FIXME: uncomment to build and save the engine
    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    engine_path = f"{args.checkpoint}/sam_{args.model_type}_embedding.trt"
    with open(engine_path, "wb") as f:
         f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}")


    # Verify the TensorRT engine
    # first get the output from pytorch model
    with torch.no_grad():
        pytorch_output = model(dummy_input).cpu().numpy()

    # then load the tensorrt engine and get the output
    TRT_LOGGER = trt.Logger()
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_buffer = {}
    input_memory = {}
    output_buffer = {}
    output_memory = {}

    for tensor in tensor_names:
        size = trt.volume(context.get_tensor_shape(tensor))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor))
        print(f"Tensor: {tensor}, Size: {size}, Dtype: {dtype}, Mode: {engine.get_tensor_mode(tensor)}")

        if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor, (1, 3, 1024, 1024))
            input_buffer[tensor] = cuda.pagelocked_empty(size, dtype)
            input_memory[tensor] = cuda.mem_alloc(input_buffer[tensor].nbytes)
            context.set_tensor_address(tensor, int(input_memory[tensor]))
        else: # OUTPUT
            output_buffer[tensor] = cuda.pagelocked_empty(size, dtype)
            output_memory[tensor] = cuda.mem_alloc(output_buffer[tensor].nbytes)
            context.set_tensor_address(tensor, int(output_memory[tensor]))

    # inference
    stream = cuda.Stream()
    np.copyto(input_buffer["input"], np.ascontiguousarray(dummy_input.cpu().numpy().astype(np.float32).ravel()))

    cuda.memcpy_htod_async(input_memory["input"], input_buffer["input"], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_buffer["image_embedding"], output_memory["image_embedding"], stream)
    stream.synchronize()
    tensorrt_output = output_buffer["image_embedding"].reshape(pytorch_output.shape)

    # compare the output
    if not np.allclose(pytorch_output, tensorrt_output, rtol=1e-02, atol=1e-02):
        print("Outputs do not match!")
    else:
        print("Outputs match!")

    # import ipdb; ipdb.set_trace()