"""levelapp/config/interaction_request.py"""
import os
import json
import yaml
import logging

from pydantic import BaseModel, HttpUrl, SecretStr, Field, computed_field
from typing import Literal, Dict, Any
from string import Template

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class EndpointConfig(BaseModel):
    """
    Configuration class for user system's endpoint.

    Parameters:
        base_url (HttpUrl): The base url of the endpoint.
        method (Literal['POST', 'GET']): The HTTP method to use (POST or GET).
        api_key (SecretStr): The API key to use.
        bearer_token (SecretStr): The Bearer token to use.
        model_id (str): The model to use (if applicable).
        payload_template (Dict[str, Any]): The payload template to use.
        variables (Dict[str, Any]): The variables to populate the payload template.

    Note:
        Either you set the environment variables providing the following:\n
        - ENDPOINT_URL="http://127.0.0.1:8000"
        - ENDPOINT_API_KEY="<API_KEY>"
        - BEARER_TOKEN="<BEARER_TOKEN>"
        - MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
        - PAYLOAD_PATH="../../src/data/payload_example_1.yaml"

        Or manually configure the model instance by assigning the proper values to the model fields.
    """
    # Required
    base_url: HttpUrl = Field(default=HttpUrl(os.getenv('ENDPOINT_URL', 'https://www.example.com')))
    method: Literal["POST", "GET"] = Field(default="POST")

    # Auth
    api_key: SecretStr = Field(default=SecretStr(os.getenv('ENDPOINT_API_KEY', '')))
    bearer_token: SecretStr | None = Field(default=SecretStr(os.getenv('BEARER_TOKEN', '')))
    model_id: str | None = Field(default=os.getenv('MODEL_ID', ''))

    # Data
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

    @computed_field()
    @property
    def full_url(self) -> str:
        return str(self.base_url)

    @computed_field()
    @property
    def headers(self) -> Dict[str, Any]:
        headers: Dict[str, Any] = {"Content-Type": "application/json"}
        if self.model_id:
            headers["x-model-id"] = self.model_id
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token.get_secret_value()}"
        if self.api_key:
            headers["x-api-key"] = self.api_key.get_secret_value()
        return headers

    @computed_field
    @property
    def payload(self) -> Dict[str, Any]:
        """Return fully prepared payload depending on template or full payload."""
        self.load_template()
        if not self.variables:
            return {}
        return self._replace_placeholders(self.payload_template, self.variables)

    @staticmethod
    def _replace_placeholders(obj: Any, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively replace placeholders in payload template with variables."""
        def _replace(_obj):
            if isinstance(_obj, str):
                subst = Template(_obj).safe_substitute(variables)
                if '$' in subst:
                    logger.warning(f"[EndpointConfig] Unsubstituted placeholder in payload:\n{subst}\n\n")
                return subst

            elif isinstance(_obj, dict):
                return {k: _replace(v) for k, v in _obj.items()}

            elif isinstance(_obj, list):
                return [_replace(v) for v in _obj]

            return _obj

        return _replace(obj)

    # TODO-0: Use 'Path' for path configuration.
    def load_template(self, path: str | None = None) -> Dict[str, Any]:
        try:
            if not path:
                path = os.getenv('PAYLOAD_PATH', '')

            if not os.path.exists(path):
                raise FileNotFoundError(f"The provide payload template file path '{path}' does not exist.")

            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    data = yaml.safe_load(f)

                elif path.endswith(".json"):
                    data = json.load(f)

                else:
                    raise ValueError("[EndpointConfig] Unsupported file format.")

                self.payload_template = data
                # TODO-1: Remove the return statement if not required.
                return data

        except FileNotFoundError as e:
            raise FileNotFoundError(f"[EndpointConfig] Payload template file '{e.filename}' not found in path.")

        except yaml.YAMLError as e:
            raise ValueError(f"[EndpointConfig] Error parsing YAML file:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"[EndpointConfig] Error parsing JSON file:\n{e}")

        except IOError as e:
            raise IOError(f"[EndpointConfig] Error reading file:\n{e}")

        except Exception as e:
            raise ValueError(f"[EndpointConfig] Unexpected error loading configuration:\n{e}")


if __name__ == '__main__':
    endpoint_config = EndpointConfig()
    print(f"Dump:\n{endpoint_config.model_dump()}")
