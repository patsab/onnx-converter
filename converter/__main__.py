import os

import torch
from colpali_engine.models import ColQwen2

from .colqwen2_model import ColQwen2Wrapper
from .dummy_inputs import dummy_inputs_for_export_tuple

og_model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0-merged")
combined_model_wrapper = ColQwen2Wrapper(og_model).eval()
onnx_path_combined = "onnx_models/colqwen2_combined.onnx"


dynamic_axes_combined = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "pixel_values": {
        0: "batch_size",
        1: "num_patches",
    },  # num_patches can be 0 for queries
    "image_grid_thw": {0: "batch_size"},
    "output_embeddings": {0: "batch_size", 1: "emb_dim"},
}


input_names_combined = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
output_names_combined = ["output_embeddings"]

print(f"\nExporting Combined encoder to {onnx_path_combined}...")
with torch.no_grad():
    torch.onnx.export(
        combined_model_wrapper,
        dummy_inputs_for_export_tuple,
        onnx_path_combined,
        input_names=input_names_combined,
        output_names=output_names_combined,
        dynamic_axes=dynamic_axes_combined,
        opset_version=20,
        verbose=False,  # Set to True for debugging export
    )
print("Combined encoder ONNX export complete.")
print(f"\nONNX model saved to '{os.path.abspath(onnx_path_combined)}'")
