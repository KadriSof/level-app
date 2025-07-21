"""levelapp/clients/openai.py"""
import os
from typing import Dict, Any

import requests
import json
from ..core.base import BaseChatClient


class OpenAIClient(BaseChatClient):
    def __init__(self, **kwargs):
        self.base_url = kwargs.get('base_url') or "https://api.openai.com/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not set")

    # TODO: Add exception handling and retry mechanism.
    def call(self, message: str, model: str, max_tokens: int = None, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "input": message
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))

        return response.json()
