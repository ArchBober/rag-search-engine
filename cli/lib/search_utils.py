import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PATH_MOVIES = os.path.join(PROJECT_ROOT, "data", "movies.json")
PATH_STOP_WORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def load_movies() -> list[dict]:
    with open(PATH_MOVIES, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[dict]:
    with open(PATH_STOP_WORDS, "r") as f:
        data = f.readlines()
    data = [line.rstrip("\n") for line in data]
    return data