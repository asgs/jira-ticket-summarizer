import hashlib
import encodings
import tiktoken
import logging
from config import settings

logger = logging.getLogger(__name__)

def gen_hash(user_input: str) -> str:
    h256 = hashlib.sha256()
    h256.update(user_input.encode(encodings.utf_8.getregentry().name))
    return h256.hexdigest()

def chunk_data(data: str, max_tokens: int = None) -> list[str]:
    if max_tokens is None or max_tokens > settings.EMBEDDING_LM_SEQ_LEN or max_tokens < 1:
        max_tokens = settings.EMBEDDING_LM_SEQ_LEN
        logger.debug(f"Resetting the sequence length to {max_tokens}")

    encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)
    tokens = encoding.encode(data)
    chunks = []
    for counter in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[counter:counter + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def build_model_prompt(contextual_data: str, query: str) -> str:
    logger.debug(f"Forming an LLM prompt using context:{contextual_data} and query:{query}")
    return f"""
        As a helpful AI assistant, your job is to 1. format and summarize the query with detailed requirements to the point using the Context provided. 2. not provide any detail outside this Context. 3. ensure your response is NOT truncated midway.

        Context: {contextual_data}

        Query: {query}

        Summary:
        """
