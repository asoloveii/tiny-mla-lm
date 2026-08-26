import os
from typing import List, Optional

from openai import OpenAI


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "tinylm")

_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop: Optional[List[str]] = None,
) -> str:
    """Plain-text completion via vLLM's /v1/completions endpoint - raw prompt in, raw text out"""
    response = _client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
    )
    return response.choices[0].text


def health_check() -> bool:
    try:
        _client.models.list()
        return True
    except Exception:
        return False
