import time
import numpy as np
import logging
from config import settings
from utils import build_model_prompt
from core.model_manager import model_manager
from core.vector_db import vector_db
from models import SummarizeRequest

logger = logging.getLogger(__name__)

class SummarizerService:
    async def summarize(self, request: SummarizeRequest):
        user_input = request.user_input
        token_count = request.token_count
        top_p = request.top_p
        temperature = request.temperature
        top_k = request.top_k
        logger.info(f"user_input is '{user_input}'")
        st = time.perf_counter()

        query_embedding = model_manager.transformer.encode(user_input)

        chunked_search_results = vector_db.query_chunks(query_embeddings=[query_embedding], n_results=top_k)
        ids = chunked_search_results['ids'][0]
        logger.debug(f"ids from chunked_docs search are {ids}")

        parent_ids = list(dict.fromkeys(id.split("_")[0] for id in ids))
        logger.debug(f"ids from chunked_docs search after deduplication are {parent_ids}")

        full_search_results = vector_db.get_full_docs(ids=parent_ids)
        logger.debug(f"full search results are {full_search_results}")
        docs = full_search_results['documents']
        logger.debug(f"Docs are {docs}")

        if not docs:
            return {
                "result": {
                    "nearest_doc": None,
                    "summary": "No relevant documents found."
                },
                "metadata": {
                    "time_taken_seconds": f"{time.perf_counter() - st:.3f}"
                }
            }

        scores = model_manager.reranker.predict([(user_input, doc) for doc in docs])
        best_doc_idx = np.argmax(scores)
        doc = docs[best_doc_idx]
        logger.debug(f"Reranked doc is {doc}")

        prompt = build_model_prompt(doc, user_input)
        in_tokens = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.causal_model.device)

        out_tokens = model_manager.causal_model.generate(
            **in_tokens,
            max_new_tokens=token_count,
            top_p=top_p,
            temperature=temperature
        )
        out_text = model_manager.tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        logger.info(f"LLM reranked Summary is {out_text}")

        summary_start = out_text.rfind(settings.SUMMARY_PREFIX)
        if summary_start != -1:
            summary = out_text[summary_start + len(settings.SUMMARY_PREFIX):].strip()
        else:
            summary = out_text

        return {
            "result": {
                "nearest_doc": doc,
                "summary": summary
            },
            "metadata": {
                "time_taken_seconds": f"{time.perf_counter() - st:.3f}"
            }
        }

summarizer_service = SummarizerService()
