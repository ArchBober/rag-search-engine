from sentence_transformers import SentenceTransformer
import numpy as np
import os
import math
import re

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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


    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

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



def fixed_size_chunking(text: str, chunk_size: int = 200) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size

    return chunks

def overlap_size_chunking(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        
        ov = 0 if i == 0 else overlap
        
        chunk_words = words[i-ov : i + chunk_size-ov]
        chunks.append(" ".join(chunk_words))
        i += chunk_size-ov

    return chunks


def chunk(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    if overlap == 0:
        chunks = fixed_size_chunking(text, chunk_size)
    else:
        chunks = overlap_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunk(text: str, chunk_size: int = 200, overlap: int = 0):
    words = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    n_words = len(words)
    i = 0
    ov = 0
    while i < n_words:
        if overlap != 0:
            ov = 0 if i == 0 else overlap
        
        chunk_words = words[i-ov : i + chunk_size-ov]
        chunks.append(" ".join(chunk_words))
        i += chunk_size-ov

    # print(f"Semantically chunking {len(text)} characters")
    # for i, chunk in enumerate(chunks):
    #     print(f"{i + 1}. {chunk}")

    return chunks
