import logging
import time
import concurrent.futures
import pandas as pd
from config import settings
from utils import gen_hash, chunk_data
from core.model_manager import model_manager
from core.vector_db import vector_db

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.futures = []

    def index_doc_into_collxn(self, data: str, index: str = None) -> None:
        logger.info(f"Model Manager is {model_manager}")
        embedding = model_manager.transformer.encode(data)
        if index is not None:
            ids = [str(index)]
        else:
            logger.info("Generating a new id for the given data")
            index = gen_hash(data)
            ids = [index]
        logger.info(f"Indexing data#{index}")
        vector_db.add_full_docs(embeddings=[embedding], ids=ids, documents=[data])

    def index_chunked_doc_into_collxn(self, row_data: str, index: str) -> None:
        chunks = chunk_data(row_data)
        chunk_count = len(chunks)
        logger.info(f"Indexing row#{index} with {chunk_count} chunk(s)")
        source_embeddings = model_manager.transformer.encode(chunks)

        ids = []
        for counter in range(chunk_count):
            ids.append(f"{index}_{counter}")
        vector_db.add_chunks(embeddings=source_embeddings, ids=ids, documents=chunks)

    def process_row(self, index, row_data) -> None:
        self.index_doc_into_collxn(data=row_data, index=index)
        self.index_chunked_doc_into_collxn(row_data=row_data, index=index)

    def ingest_from_csv(self) -> None:
        start_time = time.perf_counter()
        logger.info(f"About to read source data from the location '{settings.SRC_DATA_LOC}'")
        max_rows = settings.MAX_RECORD_COUNT;

        try:
            df = pd.read_csv(settings.SRC_DATA_LOC, nrows=max_rows)
            logger.info("Source data read successfully")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return

        with concurrent.futures.ThreadPoolExecutor(settings.MAX_THREAD_COUNT) as tp_executor:
            for index, row in df.iterrows():
                summary = row['summary']
                description = row['description']
                logger.info(f"Reading row#{index}")

                row_data = f"{summary}. {description}"

                self.futures.append(tp_executor.submit(self.index_doc_into_collxn, row_data, str(index)))
                self.futures.append(tp_executor.submit(self.index_chunked_doc_into_collxn, row_data, str(index)))

                logger.debug(f"Submitted task to create embeddings for row#{index}")
                if index == (max_rows - 1):
                    logger.info(f"Limiting ingestion to {max_rows} records")
                    break

        self._wait_for_indexing(start_time)
        del df

    async def ingest_single(self, user_input: str) -> str:
        index = gen_hash(user_input)
        self.index_doc_into_collxn(data=user_input, index=index)
        self.index_chunked_doc_into_collxn(row_data=user_input, index=index)
        return index

    def _wait_for_indexing(self, start_time) -> None:
        for future in self.futures:
            logger.debug(f"Future {future.result()} completed")
        logger.info(f"Indexing completed in {time.perf_counter() - start_time:.3f} seconds")
        self.futures = []

ingestion_service = IngestionService()
