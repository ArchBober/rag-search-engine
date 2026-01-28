
import argparse
from lib.semantic_search import semantic_chunk, chunk, search, verify_model, embed_text, verify_embeddings, embed_query_text
from lib.chunked_semantic_search import embed_chunks_command, search_chunked_command
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify Model")

    embed_txt_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_txt_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embed query")
    embedquery_parser.add_argument("embed_query_text", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="search query")
    search_parser.add_argument("embed_query_text", type=str, help="Query to embed")
    search_parser.add_argument("--limit", nargs='?', default=5, type=int, help="Limit output")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("chunk", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", nargs='?', default=200, type=int, help="Limit chunk size")
    chunk_parser.add_argument("--overlap", nargs='?', default=0, type=int, help="Overlap chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic chunk text")
    semantic_chunk_parser.add_argument("chunk", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", nargs='?', default=200, type=int, help="Limit chunk size")
    semantic_chunk_parser.add_argument("--overlap", nargs='?', default=0, type=int, help="Overlap chunk")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Semantic chunk text")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embedding = embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.embed_query_text)
        case "search":
            search(args.embed_query_text, args.limit)
        case "chunk":
            chunk(args.chunk, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.chunk, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            result = search_chunked_command(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

