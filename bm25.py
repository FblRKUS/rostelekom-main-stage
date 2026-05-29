import math
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    # Split by anything that isn't a letter or number
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


class BM25:
    def __init__(self, corpus: Optional[List[str]] = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        if corpus:
            self._initialize(corpus)

    def _initialize(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        nd: Dict[str, int] = {}
        num_doc = 0

        for document in corpus:
            self.doc_len.append(0)
            self.doc_freqs.append(defaultdict(int))

            tokens = tokenize(document)
            self.doc_len[num_doc] = len(tokens)
            self.avgdl += len(tokens)

            for word in tokens:
                self.doc_freqs[num_doc][word] += 1

            for word in self.doc_freqs[num_doc].keys():
                nd[word] = nd.get(word, 0) + 1

            num_doc += 1

        if self.corpus_size > 0:
            self.avgdl /= self.corpus_size

        for word, freq in nd.items():
            idf_score = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1)
            self.idf[word] = idf_score

    def get_scores(self, query: str) -> List[float]:
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0:
            return scores

        tokens = tokenize(query)
        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            for token in tokens:
                if token not in self.doc_freqs[i]:
                    continue

                freq = self.doc_freqs[i][token]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += self.idf.get(token, 0) * (num / den)

        return scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_freqs": [dict(df) for df in self.doc_freqs],
            "idf": self.idf,
            "doc_len": self.doc_len,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25":
        bm25 = cls()
        bm25.k1 = data["k1"]
        bm25.b = data["b"]
        bm25.corpus_size = data["corpus_size"]
        bm25.avgdl = data["avgdl"]
        bm25.doc_freqs = [defaultdict(int, df) for df in data["doc_freqs"]]
        bm25.idf = data["idf"]
        bm25.doc_len = data["doc_len"]
        return bm25
