import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        logger.info("Initializing VectorDB...")
        self.client = chromadb.Client()

        self.chunked_collection = self.client.get_or_create_collection("data-chunked")
        self.full_collection = self.client.get_or_create_collection("data-full")

    def add_full_docs(self, embeddings, ids, documents) -> None:
        self.full_collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=documents
        )

    def add_chunks(self, embeddings, ids, documents) -> None:
        self.chunked_collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=documents
        )

    def query_chunks(self, query_embeddings, n_results=5) -> dict:
        return self.chunked_collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

    def get_full_docs(self, ids) -> dict:
        return self.full_collection.get(ids=ids)

vector_db = VectorDB()
