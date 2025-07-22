"""levelapp/clients/ionos.py"""
import os
import uuid

import requests
import httpx
import json

from typing import Dict, Any
from ..core.base import BaseChatClient


class IonosClient(BaseChatClient):
    def __init__(self, **kwargs):
        self.base_url = kwargs.get('base_url')
        self.api_key = kwargs.get('api_key') or os.environ.get("IONOS_API_KEY")
        if not self.api_key:
            raise ValueError("IONOS API key not set.")

    def call(self, message: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "properties": {"input": message},
            "option": {
                "top-k": 5,
                "top-p": 0.9,
                "temperature": 0.0,
                "max_tokens": 150,
                "seed": uuid.uuid4().int & ((1 << 16) - 1),
            }
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


    async def call(self, message: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "properties": {"input": message},
            "option": {
                **self.default_options,
                **kwargs,
                "seed": uuid.uuid4().int & ((1 << 16) - 1),
            }
        }

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            return response.json()
