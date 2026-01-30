from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        img = Image.open(image_path)
        embedding = self.model.encode([img])
        return embedding[0]

def verify_image_embedding(image_path: str):
    ms = MultimodalSearch()
    print(image_path)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")