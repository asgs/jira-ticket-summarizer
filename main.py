from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import chromadb
import pandas
import tiktoken
import logging
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS

MAX_RECORD_COUNT = 100
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt="%Y-%m-%dT%H:%M:%S%z", level=logging.INFO)

logger.info("About to read source data")
dataset = pandas.read_csv("source-data/GFG_FINAL.csv", engine='pyarrow', usecols=['Summary','Description'])
logger.info("source data read successfully")

model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()

chunked_collection = chroma_client.create_collection("jira-tickets-data-chunked")
full_collection = chroma_client.create_collection("jira-tickets-data-full")

all_texts = []
all_embeddings = []

def chunk_data(data, max_tokens=512):
	#print("About to chunk the data", data);
	encoding = tiktoken.get_encoding("gpt2")
	#print("encoding in", encoding);
	tokens = encoding.encode(data)
	#print("encoded data", tokens);
	chunks = []
	for counter in range(0, len(tokens), max_tokens):
		chunk_tokens = tokens[counter:counter + max_tokens]
		chunk_text = encoding.decode(chunk_tokens)
		chunks.append(chunk_text)
	#print("chunks", chunks);
	return chunks

def index_full_collection(index, row_data):
	all_texts.append(row_data)
	row_embedding = model.encode(row_data)
	all_embeddings.append(row_embedding)
	full_collection.add(ids=[str(index)], documents=[row_data])

def index_chunked_collection(index, row_data):
	chunks = chunk_data(row_data)
	chunk_count = len(chunks)
	logger.info(f"Chunks' size is {chunk_count}")
	source_embeddings = model.encode(chunks)
	ids = []
	for counter in range(0, chunk_count):
		ids.append(str(index + counter))
	chunked_collection.add(embeddings=source_embeddings, documents=chunks, ids=ids)

# Entry point for now but to be moved out as a separate process.
for index, row in dataset.iterrows():
	summary = row['Summary']
	description = row['Description']
	logger.info(f"Indexing row# {index}")
	row_data = f"{summary}. {description}"
	index_full_collection(index, row_data)
	index_chunked_collection(index, row_data)
	if index == MAX_RECORD_COUNT:
		logger.info(f"Discontinuing the loop after {MAX_RECORD_COUNT} records")
		break
logger.info("Initializing LexRank")
lxr = LexRank(all_texts, stopwords=STOPWORDS['en'])
logger.info(f"Initialized LexRank {lxr}")

@app.post("/summarize")
async def summarize(input: str):
	logger.info(f"Input jira ticket description from user is {input}")
	embedding = model.encode(input)
	chunked_data_summary = chunked_collection.query(embedding, n_results=1)
	chunked_doc = chunked_data_summary['documents'][0]
	#print("Results are", results)
	full_data_summary = full_collection.query(embedding, n_results=1)
	full_doc = full_data_summary['documents'][0]
	full_user_summary = lxr.get_summary(full_doc, threshold=0.9)
	chunked_user_summary = lxr.get_summary(chunked_doc, threshold=0.9)
	#return {"chunked_data_summary": chunked_data_summary, "full_data_summary": full_data_summary}
	return {"chunked": {"doc":chunked_doc, "summary":chunked_user_summary}, "full": {"doc":full_doc, "summary":full_user_summary}}
