# ColQwen2 to ONNX converter

# Convert the models

To convert the models, run the following command:

```bash
python -m converter
```

This repository contains tests, which test the embeddings of the onnx models against the original models.
To run the tests, use the following command:

```bash
pytest
```

# How does the ONNX export work?

### Models

The ColPali and ColQwen2 models from HuggingFace can be loades using HF transformer.  
This implies, that the models are already `torch.nn.Module` which can be verified using `isinstance(<model>, torch.nn.Module)`.

`converter/models.py` contains wrapper classes for these models, which ovverride the forward method to accept optional parameters.
These optional parameters are only used, if an image should be embedded, not text.
Both models don't have (by default) a neutral tensor for the image parameters but rather None should be passed.
If-Operations aren't directly supported in ONNX, thus we have two variants of each model, an image embedding variant and a text embedding variant.

> **_NOTE:_** to combine both variants of a model, the torch execution graph would need to be modified to work with a neutral tensor or a tensor based if-operation.
> This is planed in the future.

### Dummy inputs

The `torch.onnx.export()` function used dummy data to trace the execution graph of the model.
Dummy data should simulate real data by being in the same shape and type as the real data.
Dynamic axes can be used, if some tensor dimensions are variable in size.  
`converter/dummy_inputs.py` creates dummy inputs in the required shape for each model variant.  
`converter/__main__.py` contains the dynamic axes for each model variant.

### Export

`converter/__main__.py` contains the conversion script.
For each model variant, the following steps are performed:

1. Init the model wrapper
2. define the dummy input
3. define the dynamic axes
4. define the input and output names
5. export the model using `torch.onnx.export()`

### OPTIONAL: Convert multiple files to single file

Since the resulting ONNX Models are greater than 2GB, they are split into multiple files.
If you want to convert this to a single file, you can use the `convert_onnx_to_single_file.py` script by the following command:

```bash
python converter/convert_onnx_to_single_file.py
```
