import httpx
from typing import List

from app.core.config import settings

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_HOST
        self.timeout = httpx.Timeout(120.0) # Local LLMs can be slow
        
    async def get_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
            
    async def generate_response(self, prompt: str, context: str = "") -> str:
        final_prompt = prompt
        if context:
            final_prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
            
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": settings.OLLAMA_GEN_MODEL, "prompt": final_prompt, "stream": False}
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]
