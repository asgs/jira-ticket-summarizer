from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import chromadb
import pandas
from pyarrow import csv
import tiktoken
import logging
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
import time
import concurrent.futures

MAX_RECORD_COUNT = 500
MAX_THREAD_COUNT = 2
SRC_DATA_LOC = "source-data/issues.csv"
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s', datefmt="%Y-%m-%dT%H:%M:%S%z", level=logging.INFO)

logger.info(f"About to read source data from {SRC_DATA_LOC}")
dataset = csv.read_csv(SRC_DATA_LOC,
	parse_options=csv.ParseOptions(newlines_in_values=True),
	read_options=csv.ReadOptions(block_size=99999999))
logger.info("source data read successfully")

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

chroma_client = chromadb.Client()

chunked_collection = chroma_client.create_collection("data-chunked")
full_collection = chroma_client.create_collection("data-full")

futures = []
all_texts = []
all_embeddings = []

def chunk_data(data, max_tokens=512):
	encoding = tiktoken.get_encoding("gpt2")
	tokens = encoding.encode(data)
	chunks = []
	for counter in range(0, len(tokens), max_tokens):
		chunk_tokens = tokens[counter:counter + max_tokens]
		chunk_text = encoding.decode(chunk_tokens)
		chunks.append(chunk_text)
	return chunks

def index_full_collection(index, row_data):
	all_texts.append(row_data)
	row_embedding = model.encode(row_data)
	all_embeddings.append(row_embedding)
	full_collection.add(ids=[str(index)], documents=[row_data])

def index_chunked_collection(index, row_data):
	chunks = chunk_data(row_data)
	chunk_count = len(chunks)
	logger.info(f"Indexing row#{index} with {chunk_count} chunk(s)")
	source_embeddings = model.encode(chunks)
	ids = []
	for counter in range(0, chunk_count):
		ids.append(str(index + counter))
	chunked_collection.add(embeddings=source_embeddings, documents=chunks, ids=ids)

def wait_for_indexing():
	for future in futures:
		logger.debug(f"Future {future.result()} completed")
	logger.info(f"Indexing completed in {time.perf_counter() - start_time} seconds")
	logger.info("Ready to serve user queries now!")

# Entry point for now but to be moved out as a separate process.
start_time = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(MAX_THREAD_COUNT) as tp_executor:
	for index, row in dataset.to_pandas().iterrows():
		summary = row['summary']
		description = row['description']
		logger.info(f"Reading row# {index}")
		row_data = f"{summary}. {description}"
		futures.append(tp_executor.submit(index_full_collection, index, row_data))
		futures.append(tp_executor.submit(index_chunked_collection, index, row_data))
		if index == (MAX_RECORD_COUNT - 1):
			logger.info(f"Discontinuing the loop after reading {MAX_RECORD_COUNT} records")
			break
logger.info("Initializing LexRank")
lxr = LexRank(all_texts, stopwords=STOPWORDS['en'])
logger.info("Initialized LexRank")
wait_for_indexing()

@app.post("/summarize")
async def summarize(input: str):
	logger.info(f"user input is {input}")
	embedding = model.encode(input)
	chunked_data_summary = chunked_collection.query(query_embeddings=[embedding], n_results=1)
	chunked_doc = chunked_data_summary['documents'][0]
	full_data_summary = full_collection.query(query_texts=[input], n_results=1)
	full_doc = full_data_summary['documents'][0]
	full_user_summary = lxr.get_summary(full_doc, threshold=0.9)
	chunked_user_summary = lxr.get_summary(chunked_doc, threshold=0.9)
	return {"chunked": {"doc":chunked_doc, "summary":chunked_user_summary}, "full": {"doc":full_doc, "summary":full_user_summary}}

