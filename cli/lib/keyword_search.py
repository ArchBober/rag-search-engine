import os
import pickle
import string
from collections import defaultdict

from nltk.stem import PorterStemmer


from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

stop_words = load_stop_words()


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")




def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def remove_stop_tokens(tokens: list[str]) -> list[str]:
    valid_tokens = []
    for token in tokens:
        if token in stop_words:
            continue
        valid_tokens.append(token)
    return valid_tokens

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    valid_tokens = []
    for token in tokens:
        valid_tokens.append(stemmer.stem(token))

    return valid_tokens



def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    tokens = remove_stop_tokens(tokens)
    tokens = stem_tokens(tokens)
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    return valid_tokens