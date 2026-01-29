import argparse

from lib.hybrid_search import (
    rrf_search_command,
)

from lib.search_utils import show_structure

from lib.augmented_generation import rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()