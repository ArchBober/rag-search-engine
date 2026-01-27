import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer


from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

stop_words = load_stop_words()

BM25_K1 = 1.5
BM25_B = 0.75


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

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
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str)-> int:
        token = tokenize_text(term)
        if len(token) != 1:
            raise Exception("term must be a single token")
        token = token[0]

        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        df = len(self.index[token])
        N = len(self.docmap)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B )-> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf*idf

    def bm25_search(self, query: str, limit: int) -> dict[int,float]:
        tokens = tokenize_text(query)
        scores: dict[int,float] = defaultdict(lambda: 0.0)
        for token in tokens:
            for doc in self.index[token]:
                scores[doc] += self.bm25(doc, token)
        
        sorted_scores = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        top_scores = {k: [v, self.docmap[k]] for k, v in sorted_scores[:limit]}
        return top_scores

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        docs_count = len(self.doc_lengths)
        if docs_count == 0:
            return 0
        total_len = sum(self.doc_lengths.values())
 
        return total_len/docs_count


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)

def bm25search_command(query: str, limit: int = 5) -> dict[int, float]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

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