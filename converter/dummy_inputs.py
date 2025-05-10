import torch
from colpali_engine.models import ColQwen2Processor
from PIL import Image
from transformers import ColPaliProcessor


def get_colqwen2_dummy_inputs_images(
    model_name: str = "vidore/colqwen2-v1.0-merged",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    processor = ColQwen2Processor.from_pretrained(model_name)
    images_raw = [
        Image.new("RGB", (8192, 8192), color="white"),
        Image.new("RGB", (2048, 1024), color="black"),
        Image.new("RGB", (1024, 2048), color="red"),
        Image.new("RGB", (512, 512), color="green"),
    ]
    dummy_images: dict[str, torch.Tensor] = dict(**processor.process_images(images_raw))
    return (
        dummy_images["input_ids"],
        dummy_images["attention_mask"],
        dummy_images["pixel_values"],
        dummy_images["image_grid_thw"],
    )


def get_colqwen2_dummy_inputs_text(
    model_name: str = "vidore/colqwen2-v1.0-merged",
) -> tuple[torch.Tensor, torch.Tensor]:
    processor = ColQwen2Processor.from_pretrained(model_name)

    queries_raw = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year’s financial performance?",
    ]
    dummy_queries: dict[str, torch.Tensor] = dict(
        **processor.process_queries(queries_raw)
    )
    return (
        dummy_queries["input_ids"],
        dummy_queries["attention_mask"],
    )


def get_colpali_dummy_inputs_images(
    model_name: str = "vidore/colpali-v1.3-hf",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processor = ColPaliProcessor.from_pretrained(model_name)
    images_raw = [
        Image.new("RGB", (8192, 8192), color="white"),
        Image.new("RGB", (1024, 1024), color="white"),
    ]
    dummy_images: dict[str, torch.Tensor] = dict(**processor.process_images(images_raw))
    return (
        dummy_images["input_ids"],
        dummy_images["attention_mask"],
        dummy_images["pixel_values"],
    )


def get_colpali_dummy_inputs_text(
    model_name: str = "vidore/colpali-v1.3-hf",
) -> tuple[torch.Tensor, torch.Tensor]:
    processor = ColPaliProcessor.from_pretrained(model_name)
    queries_raw = [
        "What is the organizational structure for our R&D department?",
        "Can you provide a breakdown of last year’s financial performance?",
    ]
    dummy_queries: dict[str, torch.Tensor] = dict(
        **processor.process_queries(queries_raw)
    )
    return (
        dummy_queries["input_ids"],
        dummy_queries["attention_mask"],
    )
