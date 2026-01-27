import logging
import time
import concurrent.futures
import pandas as pd
from config import settings
from utils import gen_hash, chunk_data
from core.model_manager import model_manager
from core.vector_db import vector_db
import numpy as np

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.futures = []

    def index_full_doc_into_collxn(self, data: str, index=None, id=None) -> None:
        # TODO - Remove this log statement once ModelManager is mulithreaded-capable.
        # logger.info(f"Model Manager is {model_manager}")
        if id is None:
            id = gen_hash(data)

        ids = [id]
        logger.info(f"Indexing data#{index} with hash {id[0:5]}")
        vector_db.add_full_docs(ids=ids, documents=[data])

    def index_chunked_doc_into_collxn(self, data: str, index=None, id=None) -> None:
        chunks = chunk_data(data)
        source_embeddings = model_manager.transformer.encode(chunks)
        ids = []
        if id is None:
            id = gen_hash(data)

        chunk_count = len(chunks)
        for counter in range(chunk_count):
            ids.append(f"{id}_{counter}")
        logger.info(f"Indexing data#{index} with hash {id[0:5]} in {chunk_count} chunk(s)")
        vector_db.add_chunks(embeddings=source_embeddings, ids=ids)

    def construct_row_data(self, row) -> str:
        cols = settings.JIRA_CSV_COLUMNS
        data_parts = []
        for col in cols:
            data_part = row[col]
            if (data_part is None or data_part == "" or data_part == "nan" or
                data_part == "NaN" or data_part is np.nan):
                logger.warn(f"Skipping unknown/invalid value '{data_part}' for the column '{col}'")
                continue
            data_parts.append(data_part)
        logger.debug(f"Data parts: {data_parts}")
        return ". ".join(data_parts)

    def ingest_from_csv(self) -> None:
        start_time = time.perf_counter()
        logger.info(f"About to read source data from the location '{settings.SRC_DATA_LOC}'")
        max_rows = settings.MAX_RECORD_COUNT;

        try:
            data_frame = pd.read_csv(settings.SRC_DATA_LOC, nrows=max_rows)
            logger.info("Source data read successfully")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return

        with concurrent.futures.ThreadPoolExecutor(settings.MAX_THREAD_COUNT) as tp_executor:
            for index, row in data_frame.iterrows():
                logger.info(f"Reading row#{index}")

                row_data = self.construct_row_data(row)

                self.futures.append(tp_executor.submit(self.index_full_doc_into_collxn, row_data, index))
                self.futures.append(tp_executor.submit(self.index_chunked_doc_into_collxn, row_data, index))

                logger.debug(f"Submitted task to create embeddings for row#{index}")
                if index == (max_rows - 1):
                    logger.info(f"Limiting ingestion to {max_rows} records")
                    break

        self._wait_for_indexing(start_time)
        del data_frame

    async def ingest_single(self, user_input: str) -> str:
        id = gen_hash(user_input)
        self.index_full_doc_into_collxn(data=user_input, id=id)
        self.index_chunked_doc_into_collxn(data=user_input, id=id)
        return id

    def _wait_for_indexing(self, start_time) -> None:
        for future in self.futures:
            logger.debug(f"Future {future.result()} completed")
        logger.info(f"Indexing completed in {time.perf_counter() - start_time:.3f} seconds")
        self.futures = []

ingestion_service = IngestionService()
