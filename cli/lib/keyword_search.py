import string
from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words

stop_words = load_stop_words()



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