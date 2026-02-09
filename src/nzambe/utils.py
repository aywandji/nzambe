from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    query_engine_loaded: bool
