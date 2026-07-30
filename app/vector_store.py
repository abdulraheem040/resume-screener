"""
FAISS-backed vector store for resume embeddings.

Keeps a parallel metadata list (resume_id, sections, raw_text) alongside
the FAISS index so search results can be mapped back to the original
resume content. For production, swap the metadata list for a real DB
(Postgres/SQLite) and store only the FAISS index + ids here.
"""

import os
import pickle
import threading

import faiss
import numpy as np


class ResumeVectorStore:
    def __init__(self, dimension: int, index_path: str = None, meta_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or "resume_index.faiss"
        self.meta_path = meta_path or "resume_meta.pkl"
        self._lock = threading.Lock()

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()
        else:
            # IndexFlatIP = exact inner-product search == cosine sim
            # since embeddings are normalized. Swap for IndexIVFFlat /
            # IndexHNSWFlat if you have >~50k resumes and need speed.
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []  # list of dicts, position matches FAISS row order

    def add(self, embedding: np.ndarray, metadata: dict) -> int:
        with self._lock:
            self.index.add(embedding.reshape(1, -1))
            self.metadata.append(metadata)
            return len(self.metadata) - 1

    def add_batch(self, embeddings: np.ndarray, metadata_list: list) -> list:
        with self._lock:
            self.index.add(embeddings)
            start = len(self.metadata)
            self.metadata.extend(metadata_list)
            return list(range(start, start + len(metadata_list)))

    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), "metadata": self.metadata[idx]})
        return results

    def save(self):
        with self._lock:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def __len__(self):
        return self.index.ntotal if hasattr(self, "index") else 0
