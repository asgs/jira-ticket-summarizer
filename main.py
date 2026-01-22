from fastapi import FastAPI
import logging
import time
from services.ingestion import ingestion_service
from services.summarizer import summarizer_service

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(threadName)s@%(name)s] - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Pre-index data from CSV on startup if needed, or this can be a separate script
    logger.info("Application starting up. Ingesting initial data...")
    ingestion_service.ingest_from_csv()
    logger.info("Ready to serve user queries now!")

@app.post("/summarize")
async def summarize(user_input: str, token_count: int = 500, top_p: float = 0.5, temperature: float = 0.7):
    return await summarizer_service.summarize(user_input, token_count, top_p, temperature)

@app.post("/ingest")
async def ingest(user_input: str):
    index = await ingestion_service.ingest_single(user_input)
    return {"status": "data ingested successfully.", "id": index}
