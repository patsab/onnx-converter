import os

import torch

from .dummy_inputs import (
    get_colpali_dummy_inputs_images,
    get_colpali_dummy_inputs_text,
    get_colqwen2_dummy_inputs_images,
    get_colqwen2_dummy_inputs_text,
)
from .models import ColPaliWrapper, ColQwen2Wrapper

ONNX_OUTPUT_PATH = "onnx_models"
os.makedirs(ONNX_OUTPUT_PATH, exist_ok=True)


# Convert ColPali-Image to ONNX
colpali_image_dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "num_tokens"},
    "attention_mask": {0: "batch_size", 1: "num_tokens"},
    "pixel_values": {0: "batch_size"},
    "output_embeddings": {0: "batch_size", 1: "num_tokens"},
}
input_names_colpali_image = ["input_ids", "attention_mask", "pixel_values"]
output_names_colpali_image = ["output_embeddings"]
with torch.no_grad():
    torch.onnx.export(
        ColPaliWrapper(),
        get_colpali_dummy_inputs_images(),
        f"{ONNX_OUTPUT_PATH}/colpali_image.onnx",
        input_names=input_names_colpali_image,
        output_names=output_names_colpali_image,
        dynamic_axes=colpali_image_dynamic_axes,
        opset_version=20,
        verbose=False,
        external_data=False,
    )

# Convert ColPali-text to ONNX
colpali_text_dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "num_tokens"},
    "attention_mask": {0: "batch_size", 1: "num_tokens"},
    "output_embeddings": {0: "batch_size", 1: "num_tokens"},
}
input_names_colpali_text = ["input_ids", "attention_mask"]
output_names_colpali_text = ["output_embeddings"]

with torch.no_grad():
    torch.onnx.export(
        ColPaliWrapper(),
        get_colpali_dummy_inputs_text(),
        f"{ONNX_OUTPUT_PATH}/colpali_text.onnx",
        input_names=input_names_colpali_text,
        output_names=output_names_colpali_text,
        dynamic_axes=colpali_text_dynamic_axes,
        opset_version=20,
        verbose=False,
        external_data=False,
    )


# Convert ColQwen2-Images to ONNX
dynamic_axes_colqwen2_images = {
    "input_ids": {0: "batch_size", 1: "num_tokens"},
    "attention_mask": {0: "batch_size", 1: "num_tokens"},
    "pixel_values": {
        0: "batch_size",
        1: "num_patches",
    },
    "image_grid_thw": {0: "batch_size"},
    "output_embeddings": {0: "batch_size", 1: "emb_dim"},
}
input_names_colqwen2_images = [
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
]
output_names_colqwen2_images = ["output_embeddings"]

with torch.no_grad():
    torch.onnx.export(
        ColQwen2Wrapper(),
        get_colqwen2_dummy_inputs_images(),
        f"{ONNX_OUTPUT_PATH}/colqwen2_image.onnx",
        input_names=input_names_colqwen2_images,
        output_names=output_names_colqwen2_images,
        dynamic_axes=dynamic_axes_colqwen2_images,
        opset_version=20,
        verbose=False,
        external_data=False,
    )

# Convert ColQwen2-Text to ONNX
dynamic_axes_colqwen2_text = dynamic_axes_colqwen2_images = {
    "input_ids": {0: "batch_size", 1: "num_tokens"},
    "attention_mask": {0: "batch_size", 1: "num_tokens"},
    "output_embeddings": {0: "batch_size", 1: "emb_dim"},
}
input_names_colqwen2_text = ["input_ids", "attention_mask"]
output_names_colqwen2_text = ["output_embeddings"]
with torch.no_grad():
    torch.onnx.export(
        ColQwen2Wrapper(),
        get_colqwen2_dummy_inputs_text(),
        f"{ONNX_OUTPUT_PATH}/colqwen2_text.onnx",
        input_names=input_names_colqwen2_text,
        output_names=output_names_colqwen2_text,
        dynamic_axes=dynamic_axes_colqwen2_text,
        opset_version=20,
        verbose=False,
        external_data=False,
    )
