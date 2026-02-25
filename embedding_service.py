from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class EmbeddingService:
    _instance: SentenceTransformer = None

    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        if cls._instance is None:
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance

    @classmethod
    def embed(cls, texts: List[str]) -> np.ndarray:
        model = cls.get_instance()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @classmethod
    def cosine_similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))