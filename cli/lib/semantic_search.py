from sentence_transformers import SentenceTransformer
import numpy as np
import os

from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("empty text")

        text_input = [text]

        embeding = self.model.encode(text_input)
        return embeding[0]

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            

        if os.path.isfile(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)

    def build_embeddings(self, documents):
        self.documents = documents
        docs_desc = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            docs_desc.append(f"{doc['title']}: {doc['description']}")
            
        self.embeddings = self.model.encode(docs_desc, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

    return embedding 


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")