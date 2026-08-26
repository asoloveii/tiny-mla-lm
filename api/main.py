import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from api import client
from api.routers import generate, rag_query


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if client.health_check():
        logger.info("vLLM backend reachable at %s", client.VLLM_BASE_URL)
    else:
        logger.warning(
            "vLLM backend NOT reachable at %s — /generate and /ask will fail until it is",
            client.VLLM_BASE_URL,
        )
    yield


app = FastAPI(title="TinyLM RAG API", lifespan=lifespan)

app.include_router(generate.router, tags=["generate"])
app.include_router(rag_query.router, tags=["rag"])


@app.get("/health")
def health():
    return {"status": "ok", "vllm_reachable": client.health_check()}
