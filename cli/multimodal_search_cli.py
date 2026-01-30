import argparse
import os

from lib.multimodal_search import verify_image_embedding
from lib.search_utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser("verify_image_embedding", help="Available commands")
    verify_image_parser.add_argument("path", type=str, help="Search query for Multimodal Search")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            path =  os.path.join(PROJECT_ROOT, args.path)
            verify_image_embedding(path)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()