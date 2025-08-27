"""levelapp/utils/loader.py"""
import json
import logging

from pathlib import Path

from collections.abc import Mapping, Sequence
from typing import Any, Type, TypeVar, List, Optional, Dict, Tuple
from pydantic import BaseModel, create_model, ValidationError

from rapidfuzz import utils


logger = logging.getLogger(__name__)
Model = TypeVar("Model", bound=BaseModel)

class DynamicModelBuilder:
    """
    Implements dynamic model builder.
    -docs here-
    """
    def __init__(self):
        self.model_cache: Dict[Tuple[str, str], Type[BaseModel]] = {}

    def clear_cache(self):
        self.model_cache.clear()

    @staticmethod
    def _sanitize_field_name(name: str) -> str:
        """
        Sanitize field names to be valid Python identifiers using rapidfuzz.
        Ensures non-empty names and handles numeric-starting names.
        """
        name = utils.default_process(name).replace(' ', '_')
        if not name:
            return "field_default"
        if name[0].isdigit():
            return f"field_{name}"
        return name

    def _get_field_type(self, value: Any, model_name: str, key: str) -> Tuple[Any, Any]:
        """
        Determine the field type and default value for a given value.
        Handles dictionaries, lists, and primitive types.
        """
        if isinstance(value, Mapping):
            nested_model = self.create_dynamic_model(model_name=f"{model_name}_{key}", data=value)
            return nested_model, ...

        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                return List[BaseModel], ...

            elif isinstance(value[0], Mapping):
                nested_model = self.create_dynamic_model(model_name=f"{model_name}_{key}", data=value[0])
                return List[nested_model], ...

            else:
                field_type = type(value[0]) if value[0] is not None else Any
                return List[field_type], ...

        else:
            field_type = Optional[type(value)] if value is not None else Optional[Any]
            return field_type, ...

    def create_dynamic_model(self, model_name: str, data: Any) -> Type[BaseModel]:
        """
        Create a Pydantic model dynamically from data.
        Supports nested dictionaries, lists, and primitives with caching.
        """
        model_name = self._sanitize_field_name(name=model_name)
        cache_key = (model_name, str(data) if not isinstance(data, dict) else str(sorted(data.keys())))

        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        if isinstance(data, Mapping):
            fields = {
                self._sanitize_field_name(name=key): self._get_field_type(value=value, model_name=model_name, key=key)
                for key, value in data.items()
            }
            model = create_model(model_name, **fields)

        else:
            field_type = Optional[type(data)] if data else Optional[Any]
            model = create_model(model_name, value=(field_type, None))

        self.model_cache[cache_key] = model

        return model


class DataLoader:
    def __init__(self):
        self.builder = DynamicModelBuilder()
        self._name = self.__class__.__name__

    @staticmethod
    def load_json_file(
            model: Type[Model],
            file_path: Path = Path("../data/conversation_example_1.json"),
    ) -> Model:
        """
        Load a JSON file and parse it into a Pydantic model instance.

        Args:
            model (Type[Model]): The Pydantic model class to instantiate.
            file_path (Path): Path to the JSON file. Defaults to 'config.json'.

        Returns:
            Model: An instance of the provided model with data from the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there's an IO error reading the file.
            ValueError: If the file contains invalid JSON.
            ValidationError: If the data doesn't validate against the model.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
            return model.model_validate(data)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {file_path}:\n{e}")

        except IOError as e:
            raise IOError(f"Error reading file {file_path}:\n{e}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}:\n{e}")

        except ValidationError:
            raise ValueError(f"Validation error while creating model {model.__name__}")

    def load_data(
            self,
            data: Dict[str, Any],
            model_name: str = "ExtractedData"
    ) -> BaseModel | None:
        """
        Load data into a dynamically created Pydantic model instance.

        Args:
            data (Dict[str, Any]): The data to load.
            model_name (str, optional): The name of the model. Defaults to "ExtractedData".

        Returns:
            An Pydantic model instance.

        Raises:
            ValidationError: If a validation error occurs.
            Exception: If an unexpected error occurs.
        """
        try:
            self.builder.clear_cache()
            dynamic_model = self.builder.create_dynamic_model(model_name=model_name, data=data)
            model_instance = dynamic_model.model_validate(data)
            return model_instance

        except ValidationError as e:
            logger.exception(f"[{self._name}] Validation Error: {e.errors()}")

        except Exception as e:
            logger.error(f"[{self._name}] An error occurred: {e}")
