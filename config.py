import os

class Settings:
    def __init__(self):
        self.MAX_RECORD_COUNT = int(os.getenv("MAX_RECORD_COUNT", 1000))
        self.MAX_THREAD_COUNT = int(os.getenv("MAX_THREAD_COUNT", 1))
        self.SRC_DATA_LOC = os.getenv("SRC_DATA_LOC", "source-data/issues.csv")

        # Model Config
        self.CAUSAL_LM_NAME = os.getenv("CAUSAL_LM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        self.EMBEDDING_LM_NAME = os.getenv("EMBEDDING_LM_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.EMBEDDING_LM_SEQ_LEN = int(os.getenv("EMBEDDING_LM_SEQ_LEN", 256))
        self.RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.TIKTOKEN_ENCODING = os.getenv("TIKTOKEN_ENCODING", "gpt2")
        self.SUMMARY_PREFIX = os.getenv("SUMMARY_PREFIX", "Summary:")

settings = Settings()
