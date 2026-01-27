from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from services.ingestion import ingestion_service
from services.summarizer import summarizer_service
from fastapi.staticfiles import StaticFiles
from request_models import SummarizeRequest, IngestRequest

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] [%(threadName)s@%(name)s] - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Stuff that only applies to Non-Production environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="frontend"), name="ui")

@app.on_event("startup")
async def startup_event():
    # Pre-index data from CSV on startup if needed, or this can be a separate script
    logger.info("Application starting up. Ingesting initial data...")
    ingestion_service.ingest_from_csv()
    logger.info("Ready to serve user queries now!")

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    return await summarizer_service.summarize(request)

@app.post("/ingest")
async def ingest(request: IngestRequest):
    index = await ingestion_service.ingest_single(request.user_input)
    return {"status": "data ingested successfully.", "id": index}
