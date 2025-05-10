import os

import onnx
from onnx.external_data_helper import convert_model_to_external_data


def convert_onnx_model_to_single_file(
    model_name: str, source_dir: str, output_dir: str
) -> None:
    """Convert an ONNX model to a single file by embedding external data into the model."""

    # Load the ONNX model
    onnx_model = onnx.load(f"{source_dir}/{model_name}")

    # Convert the model to a single file
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        size_threshold=0,
        location=f"{model_name}_data",
        convert_attribute=False,
    )

    # Save the converted model
    onnx.save(
        onnx_model,
        f"{output_dir}/{model_name}",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_name}_data",
        size_threshold=0,
    )


if __name__ == "__main__":
    SOURCE_DIR = "onnx_models"
    OUTPUT_DIR = "onnx_models_single_file"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in [
        "colpali_image.onnx",
        "colpali_text.onnx",
        "colqwen2_image.onnx",
        "colqwen2_text.onnx",
    ]:
        convert_onnx_model_to_single_file(
            model_name=model_name,
            source_dir=SOURCE_DIR,
            output_dir=OUTPUT_DIR,
        )
