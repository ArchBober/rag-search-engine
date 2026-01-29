import os
from typing import Optional
from time import sleep

from dotenv import load_dotenv
from google import genai
from .search_utils import show_structure

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def rereank_individual(query: str, results: str) -> str:
    # show_structure(results)
    for idx, doc in enumerate(results):
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 1-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        print(prompt)

        response = client.models.generate_content(model=model, contents=prompt)
        sleep(3.0)
        print(response.text)
        score = float((response.text or 0.0))
        results[idx]["metadata"]["rerank_score"] = score

        print(results[idx]["metadata"]["rerank_score"])

    results = sorted(results, key= lambda x: x["metadata"]["rerank_score"] , reverse=True)
    show_structure(results)
    return results




def rerank_results(query: str, results: str, method: Optional[str] = None) -> str:
    match method:
        case "individual":
            return rereank_individual(query, results)
        case _:
            return query