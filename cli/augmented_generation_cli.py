import argparse

from lib.hybrid_search import (
    rrf_search_command,
)

from lib.search_utils import show_structure

from lib.augmented_generation import rag, summarize, citation, question

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summ_parser = subparsers.add_parser(
        "summarize", help="Use LLM to summarise results"
    )
    summ_parser.add_argument("query", type=str, help="Search query for RAG")
    summ_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Use LLM to cite results"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    question_parser = subparsers.add_parser(
        "question", help="Use LLM to question"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query, limit=5)

            print("Search Results:")
            for doc in results['results']:
                print(f" - {doc["title"]}")

            response = rag(query, results["results"])
            print("RAG Response:")
            print(response)
        case "summarize":
            query = args.query
            limit = args.limit
            results = rrf_search_command(query, limit=limit)

            print("Search Results:")
            for doc in results['results']:
                print(f" - {doc["title"]}")

            response = summarize(query, results["results"])
            print("LLM Summary:")
            print(response)
        case "citations":
            query = args.query
            limit = args.limit
            results = rrf_search_command(query, limit=limit)

            print("Search Results:")
            for doc in results['results']:
                print(f" - {doc["title"]}")

            response = citation(query, results["results"])
            print("LLM Answer:")
            print(response)
        case "question":
            query = args.query
            limit = args.limit
            results = rrf_search_command(query, limit=limit)

            print("Search Results:")
            for doc in results['results']:
                print(f" - {doc["title"]}")

            response = question(query, results["results"])
            print("Answer:")
            print(response)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()