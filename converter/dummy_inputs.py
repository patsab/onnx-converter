from colpali_engine.models import ColQwen2Processor
from PIL import Image

processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-merged")


# dummy images
images_raw = [
    Image.new("RGB", (8192, 8192), color="white"),
    Image.new("RGB", (4096, 4096), color="white"),
]
dummy_images = dict(**processor.process_images(images_raw))
dummy_inputs_for_export_tuple = (
    dummy_images["input_ids"],
    dummy_images["attention_mask"],
    dummy_images["pixel_values"],
    dummy_images["image_grid_thw"],
)
