from .semantic_search import SemanticSearch, semantic_chunk, CACHE_DIR
from .search_utils import load_movies
import numpy as np

import os
import json

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks: List[str] = []
        metadata: List[dict] = []

        for idx, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc['description']:
                continue

            sem_chunks = semantic_chunk(doc["description"], 4, 1)
                
            chunks.extend(sem_chunks)

            for idy, sem_chunk in enumerate(sem_chunks):
               metadata.append({
                "movie_idx": doc["id"],
                "chunk_idx": idy,
                "total_chunks": len(sem_chunk)
               }) 

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            

        if os.path.isfile(self.chunk_embeddings_path) and os.path.isfile(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "rb") as f:
                self.chunk_metadata = json.load(f)
            
            return self.chunk_embeddings
            
        
        return self.build_chunk_embeddings(documents)


def embed_chunks():
    css = ChunkedSemanticSearch()
    documents = load_movies()
    chunks = css.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunks)} chunked embeddings")
