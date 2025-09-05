"""levelapp/clients/openai.py"""
import os

from typing import Dict, Any
from levelapp.core.base import BaseChatClient
from levelapp.aspects import JSONSanitizer


class OpenAIClient(BaseChatClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get('model') or "gpt-4o-mini"
        self.max_tokens = kwargs.get('max_tokens') or 1024
        self.base_url = kwargs.get('base_url') or "https://api.openai.com/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not set")

    def _endpoint(self) -> str:
        return "/chat/completions"

    def _build_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, message: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens,
        }

    def parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        sanitizer = JSONSanitizer()
        input_tokens = response.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        output = response.get("properties", {}).get("output", "")
        parsed = sanitizer.safe_load_json(text=output)
        return {"output": parsed, "metadata": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
