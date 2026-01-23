from pydantic import BaseModel

class SummarizeRequest(BaseModel):
    user_input: str
    top_k: int = 3
    token_count: int = 500
    top_p: float = 0.5
    temperature: float = 0.7

class IngestRequest(BaseModel):
    user_input: str
