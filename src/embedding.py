import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union

import httpx
from loguru import logger
from pydantic import BaseModel


class EmbeddingObject(BaseModel):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingSuccessResponse(BaseModel):
    object: Literal["list"]
    data: List[EmbeddingObject]
    model: Literal[
        "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"
    ]
    usage: Dict[str, int]


class ErrorObject(BaseModel):
    message: str
    type: str
    param: Union[None, str]
    code: Union[None, str]


class EmbeddingErrorResponse(BaseModel):
    error: ErrorObject


class OpenAIEmbeddingError(Exception):
    def __init__(self, message: str, error_type: str) -> None:
        self.message = f"OpenAI Embedding failed, with error type {error_type}, error message: *[{message}]*"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class EmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, emb_config: Dict) -> None:
        self.config = emb_config
        logger.trace(f"EMB-Initializing OpenAIEmbedding with config: {self.config}")
        # auth
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as e:
            logger.error("Can not find openai api key")
            raise ValueError("Can not find openai api key") from e
        self.header = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        with httpx.Client(timeout=self.config["embedding_timeout"]) as client:
            logger.trace(
                f"EMB-Calling OpenAIEmbedding with model: {self.config['emb_model_name']}, endpoint: {self.config['request_endpoint']}"
            )
            request_data = {
                "input": texts,
                "model": self.config["emb_model_name"],
                "encoding_format": "float",
            }

            response = client.post(
                url=self.config["request_endpoint"],
                headers=self.header,
                json=request_data,
            )

            try:
                results = EmbeddingSuccessResponse(**response.json())
                logger.trace("EMB-OpenAIEmbedding success response")
            except Exception as e:
                try:
                    error_response = EmbeddingErrorResponse(**response.json())
                    logger.error(
                        f"EMB-OpenAIEmbedding failed with error: {error_response.error.message}, error type: {error_response.error.type}"
                    )
                    raise OpenAIEmbeddingError(
                        message=error_response.error.message,
                        error_type=error_response.error.type,
                    ) from e
                except Exception:
                    response.raise_for_status()
                    logger.error("EMB-OpenAIEmbedding failed with unknown error")

            # ensure the order and return
            embeddings = sorted(results.data, key=lambda x: x.index)  # type: ignore
            return [i.embedding for i in embeddings]
