"""levelapp\clients\anthropic.py"""
import os
import requests
import json

from typing import Dict, Any
from ..core.base import BaseChatClient


class ClaudeClient(BaseChatClient):
    def __init__(self, **kwargs):
        self.model = "claude-opus-4-20250514"
        self.version = "2023-06-01"
        self.max_tokens = 1024
        self.base_url = "https://api.anthropic.com/v1"
        self.api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not set.")

    def call(self, message: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/completions"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": kwargs.get('version') or self.version,
            "content-type": "application/json"
        }
        data = {
            "model": kwargs.get('model') or self.model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": kwargs.get('max_tokens') or self.max_tokens
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
