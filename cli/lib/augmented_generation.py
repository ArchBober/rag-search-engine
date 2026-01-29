import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def rag(query: str, results):
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Skipping LLM evaluation.")
        return [0] * len(results)

    docs = ""
    for r in results:
        docs += f"Title: {r['title']}\nDocument: {r['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def summarize(query: str, results):
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Skipping LLM evaluation.")
        return [0] * len(results)
        
    docs = ""
    for r in results:
        docs += f"Title: {r['title']}\nDocument: {r['document']}\n\n"

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""


    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
    