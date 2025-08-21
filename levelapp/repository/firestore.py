"""levelapp/repository/firestore.py"""
import logging
import google.auth

from typing import List, Dict, Any, Type

from google.cloud import firestore_v1
from google.cloud.firestore_v1 import DocumentSnapshot
from google.api_core.exceptions import ClientError, ServerError, NotFound, InvalidArgument, DeadlineExceeded
from google.auth.exceptions import DefaultCredentialsError
from pydantic import ValidationError

from levelapp.core.base import BaseRepository, Model


logger = logging.getLogger(__name__)


class FirestoreRepository(BaseRepository):
    """
    Firestore implementation of BaseRepository.
    (Uses hierarchical path: {user_id}/{collection_id}/{document_id}
    """

    def __init__(self, project_id: str | Any = None, database_name: str | Any = '(default)'):
        self.project_id = project_id
        self.database_name = database_name
        self.client: firestore_v1.Client | None = None

    def connect(self) -> None:
        """
        Connects to Firestore, prioritizing the project ID passed to the constructor.
        """
        try:
            credentials, default_project_id = google.auth.default()

            if not credentials:
                raise ValueError(
                    "Failed to obtain credentials. "
                    "Please set GOOGLE_APPLICATION_CREDENTIALS "
                    "or run 'gcloud auth application-default login'."
                )

            project_id = self.project_id if self.project_id else default_project_id

            self.client = firestore_v1.Client(
                project=project_id,
                credentials=credentials,
                database=self.database_name
            )

            if not self.client:
                raise ValueError("Failed to initialize Firestore client")

            logger.info(
                f"Successfully connected to Firestore. "
                f"Project: '{self.client.project}', "
                f"Scope: '{self.client.SCOPE}'"
            )

        except (ClientError, ServerError, DefaultCredentialsError, ValueError) as e:
            logger.error(f"Failed to initialize Firestore client:\n{e}")

    def close(self) -> None:
        if self.client:
            self.client.close()

    def retrieve_document(
            self,
            user_id: str,
            collection_id: str,
            document_id: str,
            model_type: Type[Model]
    ) -> Model | None:
        """
        Retrieves a document from Firestore.

        Args:
            user_id (str): User ID.
            collection_id (str): Collection ID.
            document_id (str): Document ID.
            model_type (Type[Model]): Pydantic model for parsing.

        Returns:
            An instance of the provide Pydantic model.
        """
        if not self.client:
            logger.error("Client connection lost")
            return None

        try:
            doc_ref = self.client.collection(user_id, collection_id).document(document_id)
            snapshot: DocumentSnapshot = doc_ref.get()

            if not snapshot.exists:
                logger.warning(f"Document '{document_id}' does not exist in Firestore")
                return None

            data = snapshot.to_dict()
            return model_type.model_validate(data)

        except NotFound as e:
            logger.warning(f"Failed to retrieve Firestore document <ID:{document_id}>:\n{e}")
            return None

        except InvalidArgument as e:
            logger.error(f"Invalid argument in document path <{user_id}/{collection_id}/{document_id}>:\n{e}")
            return None

        except DeadlineExceeded as e:
            logger.error(f"Request to retrieved document <ID:{document_id}> timout:\n{e}")
            return None

        except ValidationError as e:
            logger.exception(f"Failed to parse the retrieved document <ID:{document_id}>:\n{e}")
            return None

        except Exception as e:
            logger.exception(f"Failed to retrieve Firestore document <ID:{document_id}>:\n{e}")
            return None

    def store_document(
            self,
            user_id: str,
            collection_id: str,
            document_id: str,
            data: Model
    ) -> None:
        """
        Stores a document in Firestore.

        Args:
            user_id (str): User ID.
            collection_id (str): Collection ID.
            document_id (str): Document ID.
            data (Model): An instance of the Pydantic model containing the data.
        """
        if not self.client:
            logger.error("Client connection lost")

        try:
            doc_ref = self.client.collection(user_id, collection_id).document(document_id)
            data = data.model_dump()
            doc_ref.set(data)

        except NotFound as e:
            logger.warning(f"Failed to store Firestore document <ID:{document_id}>:\n{e}")
            return None

        except InvalidArgument as e:
            logger.error(f"Invalid argument in document path <{user_id}/{collection_id}/{document_id}>:\n{e}")
            return None

        except DeadlineExceeded as e:
            logger.error(f"Request to retrieved document <ID:{document_id}> timout:\n{e}")
            return None

        except ValidationError as e:
            logger.exception(f"Failed to parse the retrieved document <ID:{document_id}>:\n{e}")
            return None

        except Exception as e:
            logger.exception(f"Failed to retrieve Firestore document <ID:{document_id}>:\n{e}")
            return None

    def query_collection(
            self,
            user_id: str,
            collection_id: str,
            filters: Dict[str, Any],
            model_type: Type[Model]
    ) -> List[Model]:
        """
        Queries a collection with specified filters.

        Args:
            user_id: The ID of the user.
            collection_id: The ID of the nested collection.
            filters: A dictionary of key-value pairs to filter the query.
            model_type: The class to deserialize the documents into.

        Returns:
            A list of deserialized models that match the query.
        """
        if not self.client:
            logger.error("Client connection lost")
            return []

        try:
            collection_ref = self.client.collection(user_id, collection_id)
            query = collection_ref

            for key, value in filters.items():
                query = query.where(key, "==", value)

            results = []
            for doc in query.stream():
                if doc.exists and doc.to_dict():
                    results.append(model_type.model_validate(doc.to_dict()))

            return results

        except NotFound as e:
            logger.warning(f"Collection for user '{user_id}' not found:\n{e}")
            return []

        except InvalidArgument as e:
            logger.error(f"Invalid query argument for user '{user_id}':\n{e}")
            return []

        except DeadlineExceeded as e:
            logger.error(f"Query for user '{user_id}' timed out:\n{e}")
            return []

        except ValidationError as e:
            logger.exception(f"Failed to parse a document from query results:\n{e}")
            return []

        except Exception as e:
            logger.exception(f"An unexpected error occurred during collection query:\n{e}")
            return []

    def delete_document(
            self,
            user_id: str,
            collection_id: str,
            document_id: str
    ) -> bool:
        """
        Deletes a document from Firestore.

        Args:
            user_id: The ID of the user.
            collection_id: The ID of the nested collection.
            document_id: The ID of the document to delete.

        Returns:
            True if the document was deleted successfully, False otherwise.
        """
        if not self.client:
            logger.error("Client connection lost")
            return False

        try:
            doc_ref = self.client.collection(user_id, collection_id).document(document_id)
            doc_ref.delete()
            logger.info(f"Document '{document_id}' deleted successfully.")
            return True

        except NotFound as e:
            logger.warning(f"Failed to delete document. Document '{document_id}' not found:\n{e}")
            return False
        except InvalidArgument as e:
            logger.error(f"Invalid argument in document path <{user_id}/{collection_id}/{document_id}>:\n{e}")
            return False
        except DeadlineExceeded as e:
            logger.error(f"Request to delete document <ID:{document_id}> timed out:\n{e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to delete Firestore document <ID:{document_id}>:\n{e}")
            return False
