"""levelapp/clients/mistral.py"""
import os
import requests
import json

from typing import Dict, Any
from ..core.base import BaseChatClient


class MistralClient(BaseChatClient):
    def __init__(self, **kwargs):
        self.model = "mistral-large-latest"
        self.base_url = "https://api.mistral.ai/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Missing API key not set.")

    def call(self, message: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": kwargs.get("model") or self.model,
            "messages": [{"role": "user", "content": message}],
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected error occurred: {req_err}")
            raise

    async def acall(self, message: str, **kwargs) -> Dict[str, Any]:
        pass
