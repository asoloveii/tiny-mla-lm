from fastapi import APIRouter, HTTPException

from api import client
from api.schemas import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        text = client.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc
    return GenerateResponse(text=text)
