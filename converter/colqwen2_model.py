from typing import Optional

import torch
import torch.nn as nn
from colpali_engine.models import ColQwen2
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


class ColQwen2ImageProcessor(nn.Module):
    """
    ONNX-compatible nn.Module for processing images with Qwen2VL.

    This module is designed to handle only the image processing part of the ColQwen2Processor,
    specifically the process_images method converted to a forward method.
    """

    visual_prompt_prefix = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    image_token = "<|image_pad|>"

    def __init__(self, processor=None):
        """
        Initialize the ColQwen2ImageProcessor.

        Args:
            processor: An instance of Qwen2VLProcessor
        """
        super().__init__()
        self.processor = processor
        if self.processor is not None:
            self.processor.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device_map=None, **kwargs):
        """
        Create a ColQwen2ImageProcessor from a pretrained model.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model or its name
            device_map: Device mapping strategy
            **kwargs: Additional arguments to pass to the Qwen2VLProcessor

        Returns:
            An instance of ColQwen2ImageProcessor
        """
        processor = Qwen2VLProcessor.from_pretrained(
            pretrained_model_name_or_path, device_map=device_map, **kwargs
        )

        instance = cls(processor=processor)

        # Handle max_num_visual_tokens if provided
        if "max_num_visual_tokens" in kwargs:
            instance.processor.image_processor.max_pixels = (
                kwargs["max_num_visual_tokens"] * 28 * 28
            )
            instance.processor.image_processor.size["longest_edge"] = (
                instance.processor.image_processor.max_pixels
            )

        return instance

    @property
    def image_token_id(self) -> int:
        return self.processor.tokenizer.convert_tokens_to_ids(self.image_token)

    def forward(
        self, images: list[Image.Image], context_prompts: list[str] | None = None
    ) -> BatchFeature:
        """
        Forward method for processing images (previously process_images method).

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. text descriptions of the image context.

        Returns:
            BatchFeature: Processed batch with pixel_values and other necessary fields
        """
        if context_prompts:
            if len(images) != len(context_prompts):
                raise ValueError("Length of images and context prompts must match.")
            texts_doc = context_prompts
        else:
            texts_doc = [self.visual_prompt_prefix] * len(images)

        images = [image.convert("RGB") for image in images]

        batch_doc = self.processor(
            text=texts_doc,
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = (
            batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        )  # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        return batch_doc

    def get_n_patches(self, image_size, spatial_merge_size):
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        Args:
            image_size: Tuple of (height, width)
            spatial_merge_size: Number of patches to merge spatially

        Returns:
            Tuple of (n_patches_x, n_patches_y)
        """
        patch_size = self.processor.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.size["shortest_edge"],
            max_pixels=self.processor.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask that identifies the image tokens in the batch.

        Args:
            batch_images: BatchFeature from processed images

        Returns:
            torch.Tensor: Boolean mask identifying image tokens
        """
        return batch_images.input_ids == self.image_token_id


# class ColQwen2ModelWrapper(nn.Module):
#     og_model: nn.Module
#     og_processor: nn.Module
#
#     def __init__(self):
#         super().__init__()
#         self.model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0-merged")
#         self.processor = ColQwen2ImageProcessor.from_pretrained(
#             "vidore/colqwen2-v1.0-merged"
#         )
#
#     def forward(self):
#         batch_images = self.processor()


class ColQwen2Wrapper(nn.Module):
    def __init__(self, model: ColQwen2):
        super().__init__()
        self.model = model.eval()  # Ensure model is in eval mode

    def forward(
        self,
        input_ids: torch.LongTensor,  # exp shape (B, num_tokens)
        attention_mask: torch.Tensor,  # Exp shape (B, num_tokens)
        pixel_values: Optional[torch.Tensor] = None,  # (B, num_patches, 1176)
        image_grid_thw: Optional[torch.LongTensor] = None,  # Exp shape: (B, 3)
    ):
        if pixel_values is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        if image_grid_thw is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
