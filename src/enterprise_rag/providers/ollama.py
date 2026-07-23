"""Ollama adapters using only the Python standard library HTTP stack."""
from __future__ import annotations

import json
import socket
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib import error, request

from enterprise_rag.embeddings import EmbeddingValidationError, validate_embedding_vectors
from enterprise_rag.generation import GenerationValidationError, validate_generated_text

class OllamaError(RuntimeError):
    """Base error for Ollama transport and response failures."""

class OllamaConnectionError(OllamaError): pass
class OllamaTimeoutError(OllamaError): pass
class OllamaHTTPError(OllamaError): pass
class OllamaResponseError(OllamaError): pass

class JsonTransport(Protocol):
    def post_json(self, url: str, payload: Mapping[str, Any], timeout: float) -> Any: ...

class UrllibJsonTransport:
    """Small synchronous JSON transport for local Ollama HTTP calls."""
    def post_json(self, url: str, payload: Mapping[str, Any], timeout: float) -> Any:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                raw = response.read()
        except error.HTTPError as exc:
            raise OllamaHTTPError(f"Ollama returned HTTP status {exc.code} for {url}.") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {timeout} seconds: {url}.") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(f"Ollama request timed out after {timeout} seconds: {url}.") from exc
            raise OllamaConnectionError(f"Could not connect to Ollama at {url}.") from exc
        if not 200 <= int(status) < 300:
            raise OllamaHTTPError(f"Ollama returned HTTP status {status} for {url}.")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaResponseError(f"Ollama returned invalid JSON from {url}.") from exc

class OllamaEmbeddingClient:
    def __init__(
        self, *, base_url: str, model: str, timeout: float, transport: JsonTransport | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("OLLAMA_EMBEDDING_MODEL must not be empty.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than 0.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport or UrllibJsonTransport()

    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        result = self._transport.post_json(
            f"{self.base_url}/api/embed",
            {"model": model, "input": list(contents)},
            self.timeout,
        )
        if not isinstance(result, dict) or not isinstance(result.get("embeddings"), list):
            raise OllamaResponseError("Ollama embedding response must contain an embeddings array.")
        try:
            return validate_embedding_vectors(result["embeddings"], expected_count=len(contents))
        except (TypeError, ValueError, EmbeddingValidationError) as exc:
            if isinstance(exc, EmbeddingValidationError):
                raise
            raise OllamaResponseError("Ollama embedding response contains invalid vector values.") from exc

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return validate_embedding_vectors(
            self.embed(model=self.model, contents=texts), expected_count=len(texts)
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_documents([text])[0]

class OllamaGenerationClient:
    def __init__(
        self, *, base_url: str, model: str, timeout: float, transport: JsonTransport | None = None
    ) -> None:
        if not model.strip():
            raise ValueError("OLLAMA_CHAT_MODEL must not be empty.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than 0.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport or UrllibJsonTransport()

    def generate(self, prompt: str) -> str:
        result = self._transport.post_json(
            f"{self.base_url}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False},
            self.timeout,
        )
        if not isinstance(result, dict) or "response" not in result:
            raise OllamaResponseError("Ollama generation response must contain a response field.")
        try:
            return validate_generated_text(result["response"])
        except GenerationValidationError:
            raise