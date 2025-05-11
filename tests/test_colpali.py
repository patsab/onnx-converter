from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

TEST_DIR = Path(__file__).parent
ONNX_MODEL_PATH = Path(__file__).parent.parent


def test_colpali_image_model() -> None:
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
    images = [
            Image.new("RGB", (8192, 8192), color="white"),
            Image.new("RGB", (2048, 1024), color="black"),
            ]
    image_processed = processor(images=images)

    hf_model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf").eval()
    with torch.no_grad():
        hf_output = hf_model(**image_processed).embeddings

    onnx_model = ort.InferenceSession(
        (ONNX_MODEL_PATH / "colpali_image_onnx" / "colpali_image.onnx")
    )
    onnx_output = onnx_model.run(
        ["output_embeddings"],
        {
            "input_ids": image_processed["input_ids"].numpy(),
            "attention_mask": image_processed["attention_mask"].numpy(),
            "pixel_values": image_processed["pixel_values"].numpy(),
        },
    )[0]

    assert hf_output.shape == onnx_output.shape
    # calculate mean diff to check if the embeddings are similar
    mean_diff = np.mean(np.abs(onnx_output - hf_output.numpy()))
    assert mean_diff < 1e-4, f"Mean difference: {mean_diff}"


def test_colpali_text_model() -> None:
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
    text_processed = processor(
        text=["What is the organizational structure for our R&D department?"]
    )

    hf_model = ColPaliForRetrieval.from_pretrained("vidore/colpali-v1.3-hf").eval()
    with torch.no_grad():
        hf_output = hf_model(**text_processed).embeddings

    onnx_model = ort.InferenceSession(
        (ONNX_MODEL_PATH / "colpali_text_onnx" / "colpali_text.onnx")
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
