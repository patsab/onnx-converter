from fastembed import LateInteractionMultimodalEmbedding
from PIL import Image


model = LateInteractionMultimodalEmbedding(model_name="Qdrant/colpali-v1.3-fp16")


query = "What is Qdrant?"
query_embedding = model.embed_text(query)
print(next(query_embedding))


img = Image.new("RGB", (128, 128), color="white")
img_embedding = model.embed_image(img)
print(next(img_embedding))
