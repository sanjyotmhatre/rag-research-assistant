from dataclasses import dataclass
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str

def split_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]

class LocalRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("At least one chunk is required")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not query.strip():
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

def load_folder(folder: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(folder.glob("*.txt")):
        for i, part in enumerate(split_text(path.read_text(encoding="utf-8"))):
            chunks.append(Chunk(path.name, i, part))
    return chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("question")
    args = parser.parse_args()
    retriever = LocalRetriever(load_folder(args.folder))
    for chunk, score in retriever.search(args.question):
        print(f"[{chunk.source}#{chunk.index} score={score:.3f}] {chunk.text}\n")
