from typing import Optional

import torch
import torch.nn as nn
from colpali_engine.models import ColPali, ColQwen2
from transformers import ColPaliForRetrieval


class ColQwen2Wrapper(nn.Module):
    model: ColQwen2

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0-merged"):
        super().__init__()
        self.model = ColQwen2.from_pretrained(model_name).eval()

    def forward(
        self,
        input_ids: torch.LongTensor,  # exp shape (B, num_tokens)
        attention_mask: torch.LongTensor,  # Exp shape (B, num_tokens)
        pixel_values: Optional[torch.Tensor] = None,  # (B, num_patches, 1176)
        image_grid_thw: Optional[torch.LongTensor] = None,  # Exp shape: (B, 3)
    ) -> torch.Tensor:
        if pixel_values is not None and image_grid_thw is not None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class ColPaliWrapper(nn.Module):
    model: ColPali

    def __init__(self, model_name: str = "vidore/colpali-v1.3-hf"):
        super().__init__()
        self.model = ColPaliForRetrieval.from_pretrained(model_name, torch_dtype=torch.float32).eval()

    def forward(
        self,
        input_ids: torch.LongTensor,  # exp shape (B, num_tokens)
        attention_mask: torch.Tensor,  # exp shape (B, num_tokens)
        pixel_values: Optional[torch.Tensor] = None,  # exp shape (B, 3, 448, 448)
    ) -> torch.Tensor:  # exp shape (B, num_tokens, 128)
        if pixel_values is None:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).embeddings
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        ).embeddings
