from sentence_transformers import SentenceTransformer



class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def build(self) -> None:
        return

    def save(self) -> None:
        return

    def load(self) -> None:
        return

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")