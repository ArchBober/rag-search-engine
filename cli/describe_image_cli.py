import argparse
import mimetypes

from lib.hybrid_search import (
    rrf_search_command,
)

import os

from dotenv import load_dotenv
from google import genai

from google.genai import types

from lib.search_utils import BEAR_IMAGE_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument("--image", type=str, help="Search query for RAG")
    parser.add_argument("--query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    query = args.query
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Skipping LLM evaluation.")
        return

    with open(BEAR_IMAGE_PATH, "rb") as f:
        img = f.read()

    prompt = f"""
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""

    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()
           
