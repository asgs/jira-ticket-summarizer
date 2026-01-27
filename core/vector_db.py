from torch.utils.hipify.hipify_python import meta_data
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        logger.info("Initializing VectorDB...")
        self.client = chromadb.PersistentClient()
        meta_data = {"hnsw:space": "cosine",
            "hnsw:construction_ef": 1000,
            "hnsw:M": 160,
            "hnsw:search_ef": 100,
            "hnsw:batch_size": 1000,
            "hnsw:sync_threshold": 2000}
        self.chunked_collection = self.client.get_or_create_collection("data-chunked", metadata=meta_data)
        self.full_collection = self.client.get_or_create_collection("data-full", metadata=meta_data)

    def add_full_docs(self, ids, documents) -> None:
        self.full_collection.add(
            ids=ids,
            documents=documents
        )

    def add_chunks(self, embeddings, ids) -> None:
        self.chunked_collection.add(
            embeddings=embeddings,
            ids=ids
        )

    def query_chunks(self, query_embeddings, n_results=5) -> dict:
        return self.chunked_collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

    def get_full_docs(self, ids) -> dict:
        return self.full_collection.get(ids=ids)

vector_db = VectorDB()
