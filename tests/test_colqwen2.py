from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from colpali_engine import ColQwen2, ColQwen2Processor
from PIL import Image

ONNX_MODEL_PATH = Path(__file__).parent.parent


def test_colqwen2_image_model() -> None:
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-merged")
    images = [
            Image.new("RGB", (8192, 8192), color="white"),
            Image.new("RGB", (2048, 1024), color="black"),
            ]
    image_processed: dict[str, torch.Tensor] = dict(**processor.process_images(images))

    hf_model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0-merged").eval()
    with torch.no_grad():
        hf_output = hf_model(**image_processed)

    onnx_model = ort.InferenceSession(
        (ONNX_MODEL_PATH / "colqwen2_image_onnx" / "colqwen2_image.onnx")
    )
    onnx_output = onnx_model.run(
        ["output_embeddings"],
        {
            "input_ids": image_processed["input_ids"].numpy(),
            "attention_mask": image_processed["attention_mask"].numpy(),
            "pixel_values": image_processed["pixel_values"].numpy(),
            "image_grid_thw": image_processed["image_grid_thw"].numpy(),
        },
    )[0]
    # assert False
    assert hf_output.shape == onnx_output.shape
    # calculate mean diff to check if the embeddings are similar
    mean_diff = np.mean(np.abs(onnx_output - hf_output.numpy()))
    assert mean_diff < 1e-4, f"Mean difference: {mean_diff}"


def test_colqwen2_text_model() -> None:
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-merged")
    text_processed = processor.process_queries(
        ["What is the organizational structure for our R&D department?"]
    )

    hf_model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0-merged").eval()
    with torch.no_grad():
        hf_output = hf_model(**text_processed)

    onnx_model = ort.InferenceSession(
        (ONNX_MODEL_PATH / "colqwen2_text_onnx" / "colqwen2_text.onnx")
    )
    onnx_output = onnx_model.run(
        ["output_embeddings"],
        {
            "input_ids": text_processed["input_ids"].numpy(),
            "attention_mask": text_processed["attention_mask"].numpy(),
        },
    )[0]

    assert hf_output.shape == onnx_output.shape
    # calculate mean diff to check if the embeddings are similar
    mean_diff = np.mean(np.abs(onnx_output - hf_output.numpy()))
    assert mean_diff < 1e-4, f"Mean difference: {mean_diff}"
