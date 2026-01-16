from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chromadb
import pandas
from pyarrow import csv
import tiktoken
import logging
#from lexrank import LexRank
#from lexrank.mappings.stopwords import STOPWORDS
import time
import concurrent.futures
import hashlib
import encodings

MAX_RECORD_COUNT = 1000
MAX_THREAD_COUNT = 8
SRC_DATA_LOC = "source-data/issues.csv"
CAUSAL_LM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"#"Qwen/Qwen2.5-1.5B-Instruct"
#CAUSAL_LM_NAME = "ibm-granite/granite-4.0-h-350m"#"HuggingFaceTB/SmolLM2-135M-Instruct"
EMBEDDING_LM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_PREFIX = "Summary:"
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(threadName)s/%(name)s] - %(message)s', level=logging.INFO)

logger.info(f"About to read source data from {SRC_DATA_LOC}")
dataset = csv.read_csv(SRC_DATA_LOC,
	parse_options=csv.ParseOptions(newlines_in_values=True),
	read_options=csv.ReadOptions(block_size=99999999))
logger.info("source data read successfully")

transformer = SentenceTransformer(EMBEDDING_LM_NAME)

tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM_NAME)
auto_causal_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_NAME,
	device_map="auto")
#auto_causal_model.max_seq_length = 2000
chroma_client = chromadb.Client()

#chunked_doc_collxn = chroma_client.create_collection("data-chunked")
full_doc_collxn = chroma_client.create_collection("data-full")

futures = []
#all_texts = []
#all_embeddings = []

def chunk_data(data, max_tokens=512):
	encoding = tiktoken.get_encoding("gpt2")
	tokens = encoding.encode(data)
	chunks = []
	for counter in range(0, len(tokens), max_tokens):
		chunk_tokens = tokens[counter:counter + max_tokens]
		chunk_text = encoding.decode(chunk_tokens)
		chunks.append(chunk_text)
	return chunks

def gen_hash(input: str) -> str:
	h256 = hashlib.sha256()
	h256.update(input.encode(encodings.utf_8.getregentry().name))
	return h256.hexdigest()

def index_full_doc_into_collxn(data, index=None):
	#all_texts.append(data)
	embedding = transformer.encode(data)
	#all_embeddings.append(embedding)
	if index != None:
		logger.info(f"Indexing data#{index}'s full contents as is")
		full_doc_collxn.add(embeddings=[embedding], ids=[str(index)], documents=[data])
	else:
		logger.info(f"Indexing new data's full contents as is")
		full_doc_collxn.add(embeddings=[embedding], ids=[gen_hash(data)], documents=[data])

def index_chunked_doc_into_collxn(index, row_data):
	chunks = chunk_data(row_data)
	chunk_count = len(chunks)
	logger.info(f"Indexing row#{index} with {chunk_count} chunk(s)")
	source_embeddings = transformer.encode(chunks)
	if index == None:
		logger.info("Autogenerating id for this data")
		index = gen_hash(row_data)
	ids = []
	for counter in range(0, chunk_count):
		ids.append(f"{index}_{counter}")
	chunked_doc_collxn.add(embeddings=source_embeddings, documents=chunks, ids=ids)

def wait_for_indexing():
	for future in futures:
		logger.debug(f"Future {future.result()} completed")
	logger.info(f"Indexing completed in {time.perf_counter() - start_time:.3f} seconds")

# Entry point for now but to be moved out as a separate process.
start_time = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(MAX_THREAD_COUNT) as tp_executor:
	for index, row in dataset.to_pandas().iterrows():
		summary = row['summary']
		description = row['description']
		logger.info(f"Reading row# {index}")
		# Summary and Description capture the essence of data involved.
		row_data = f"{summary}. {description}"
		futures.append(tp_executor.submit(index_full_doc_into_collxn, row_data, index))
		#futures.append(tp_executor.submit(index_chunked_doc_into_collxn, index, row_data))
		if index == (MAX_RECORD_COUNT - 1):
			logger.info(f"Limiting the ingesiton to {MAX_RECORD_COUNT} records")
			break
wait_for_indexing()

logger.info("Deleting the pyarrow table")
del dataset
#logger.info(f"Initializing LexRank with {len(all_texts)} sentences")
#lxr = LexRank(all_texts, stopwords=STOPWORDS['en'])
#logger.info("Initialized LexRank")
logger.info("Ready to serve user queries now!")

@app.post("/summarize")
async def summarize(input: str):
	logger.info(f"user input is '{input}'")
	st = time.perf_counter()
	query_embedding = transformer.encode(input)
	full_data_summary = full_doc_collxn.query(query_embeddings=[query_embedding], n_results=1)
	full_doc = full_data_summary['documents'][0]
	input_tokens = tokenizer(build_model_prompt(full_doc[0], input), return_tensors="pt").to(auto_causal_model.device)
	output_tokens = auto_causal_model.generate(**input_tokens, max_new_tokens=1000, top_p=0.9, temperature=0.7)
	output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
	print(f"Summary is {output_text}")
	return {
		"result": {
			"nearest_doc": full_doc,
			"summary": output_text[output_text.find(SUMMARY_PREFIX):]
		},
		"metadata": {
			"time_taken_seconds": f"{time.perf_counter() - st:.3f}"
		}
	}

def build_model_prompt(contextual_data: str, query: str):
	return f"""
		You are a helpful assistant. You provide your response to the point without asking any follow-up questions unless absolutely necessary.

		Context: {contextual_data}

		Query: {query}

		Summary:
		"""

@app.post("/ingest")
async def ingest(input: str):
	index_full_doc_into_collxn(data=input)

