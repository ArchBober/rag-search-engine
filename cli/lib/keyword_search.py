from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    preprocessed_query = preprocess_text(query)

    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if len(results) >= limit:
            break
        for token in preprocessed_query:
            if token in preprocessed_title:
                results.append(movie)
                break

    return results

def preprocess_text(text: str) -> List(str):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text_tokens = " ".join(text.split()).split(" ")
    return text_tokens