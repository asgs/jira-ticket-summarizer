import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import settings
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    logger.info("ModelManager...")
    _instance = None

    def __new__(cls):
        logger.info("Newing ModelManager...")
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        logger.info("Initializing ModelManager...")
        if self.initialized:
            return

        self._transformer = None
        self._reranker = None
        self._tokenizer = None
        self._causal_model = None
        self.initialized = True

    @property
    def transformer(self):
        if self._transformer is None:
            logger.info(f"Loading SentenceTransformer: {settings.EMBEDDING_LM_NAME}")
            self._transformer = SentenceTransformer(settings.EMBEDDING_LM_NAME)
        return self._transformer

    @property
    def reranker(self):
        if self._reranker is None:
            logger.info(f"Loading CrossEncoder: {settings.RERANKER_MODEL_NAME}")
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        return self._reranker

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            logger.info(f"Loading Tokenizer: {settings.CAUSAL_LM_NAME}")
            self._tokenizer = AutoTokenizer.from_pretrained(settings.CAUSAL_LM_NAME)
        return self._tokenizer

    @property
    def causal_model(self):
        if self._causal_model is None:
            logger.info(f"Loading Causal Model: {settings.CAUSAL_LM_NAME}")
            self._causal_model = AutoModelForCausalLM.from_pretrained(
                settings.CAUSAL_LM_NAME,
                device_map="auto",
                revision=settings.CAUSAL_LM_REVISION
            )
        return self._causal_model

logger.info("Preparing ModelManager...")
model_manager = ModelManager()
