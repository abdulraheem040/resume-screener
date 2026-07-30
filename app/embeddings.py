"""
Embedding generation using Sentence Transformers.

Wraps model loading + encoding so the rest of the app never touches
the model directly. Embeddings are L2-normalized so that inner-product
search in FAISS is equivalent to cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"  # 768-dim, good quality/speed tradeoff


class EmbeddingModel:
    _instance = None

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbeddingModel":
        # Simple singleton so we don't reload the model on every request
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # -> cosine similarity via inner product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")
