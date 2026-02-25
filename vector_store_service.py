import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from services.embedding_service import EmbeddingService
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class VectorStoreService:
    _client: chromadb.Client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=CHROMA_DIR)
        return cls._client

    @classmethod
    def get_or_create_collection(cls, name: str):
        client = cls.get_client()
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def store_chunks(cls, collection_name: str, chunks: List[Dict]) -> int:
        """Store chunks with their embeddings into ChromaDB."""
        collection = cls.get_or_create_collection(collection_name)

        # Clear existing docs in collection
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        texts = [c["text"] for c in chunks]
        embeddings = EmbeddingService.embed(texts).tolist()

        collection.add(
            ids=[f"chunk_{c['id']}" for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "chunk_id": c["id"],
                "token_count": c["token_count"],
                "strategy": c.get("metadata", {}).get("strategy", "unknown"),
                **{k: str(v) for k, v in c.get("metadata", {}).items()}
            } for c in chunks]
        )
        return len(chunks)

    @classmethod
    def retrieve(cls, collection_name: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find top-k most relevant chunks for a query."""
        collection = cls.get_or_create_collection(collection_name)
        query_embedding = EmbeddingService.embed([query])[0].tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": round(1 - results["distances"][0][i], 4)
            })
        return retrieved

    @classmethod
    def list_collections(cls) -> List[str]:
        return [c.name for c in cls.get_client().list_collections()]

    @classmethod
    def delete_collection(cls, name: str):
        cls.get_client().delete_collection(name)