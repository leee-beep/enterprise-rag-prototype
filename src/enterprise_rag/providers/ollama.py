"""Ollama adapters using only the Python standard library HTTP stack."""
from __future__ import annotations

import json
import socket
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from urllib import error, request

from enterprise_rag.embeddings import EmbeddingValidationError, validate_embedding_vectors
from enterprise_rag.generation import (
    GenerationValidationError,
    validate_generated_text,
    validate_generation_prompt,
)

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
            detail = _http_error_detail(exc, payload)
            suffix = f" Error: {detail}" if detail else ""
            raise OllamaHTTPError(
                f"Ollama returned HTTP status {exc.code} for {url}.{suffix}"
            ) from exc
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

def _http_error_detail(
    exc: error.HTTPError,
    payload: Mapping[str, Any],
    *,
    maximum_length: int = 200,
) -> str | None:
    """Extract a short Ollama error without exposing request text or payloads."""
    try:
        raw = exc.read()
        parsed = json.loads(raw.decode("utf-8"))
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("error"), str):
        return None

    detail = " ".join(parsed["error"].split())
    sensitive_values: list[str] = []
    for key in ("input", "prompt"):
        value = payload.get(key)
        if isinstance(value, str):
            sensitive_values.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            sensitive_values.extend(item for item in value if isinstance(item, str))
    for sensitive in sorted(sensitive_values, key=len, reverse=True):
        if sensitive:
            detail = detail.replace(sensitive, "[redacted input]")
    if not detail:
        return None
    if len(detail) > maximum_length:
        return detail[: maximum_length - 3].rstrip() + "..."
    return detail


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        batch_size: int = 32,
        transport: JsonTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("OLLAMA_EMBEDDING_MODEL must not be empty.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be greater than 0.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Ollama embedding batch size must be a positive integer.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._transport = transport or UrllibJsonTransport()

    def embed(self, *, model: str, contents: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(contents), self.batch_size):
            batch = contents[start : start + self.batch_size]
            result = self._transport.post_json(
                f"{self.base_url}/api/embed",
                {"model": model, "input": list(batch)},
                self.timeout,
            )
            vectors.extend(self._validate_batch(result, expected_count=len(batch)))
        return validate_embedding_vectors(vectors, expected_count=len(contents))

    @staticmethod
    def _validate_batch(result: Any, *, expected_count: int) -> tuple[tuple[float, ...], ...]:
        if not isinstance(result, dict) or not isinstance(result.get("embeddings"), list):
            raise OllamaResponseError("Ollama embedding response must contain an embeddings array.")
        try:
            return validate_embedding_vectors(
                result["embeddings"], expected_count=expected_count
            )
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

    @property
    def provider(self) -> str:
        return "ollama"

    def generate(self, prompt: str) -> str:
        prompt = validate_generation_prompt(prompt)
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
