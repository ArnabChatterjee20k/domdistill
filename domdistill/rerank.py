"""Cross-encoder rerankers for chunk selection.

The default selection path in :mod:`domdistill.selection` is a bi-encoder: it
embeds the query, the heading and every candidate chunk independently and ranks
by cosine similarity. A reranker instead reads the query and a candidate chunk
*together* and emits a single relevance score, which is usually more accurate at
the cost of one model forward pass per candidate.

Two batteries-included adapters frame relevance as a ``noul`` (yes/no) decision —
"is this passage relevant to the query?" — and use ``P(true)`` as the score:

* ``LayaReranker`` wires in `laya <https://github.com/NandhaKishorM/laya>`_, an
  in-process non-autoregressive decision engine. Optional dependency; install
  with ``pip install 'domdistill[laya]'``.
* ``Tev1Reranker`` calls the same style of decision model served by
  `Ollama <https://ollama.com/library/tev1>`_ over its ``/v1/systemone``
  endpoint (default model ``tev1:0.8b``, ~800 MB, no torch). Needs only a
  running Ollama server — no extra Python dependency.

Both implement the ``RerankFn`` signature ``(query, heading, candidates) ->
list[float]`` so either can be passed as ``rerank_fn=`` to the chunker.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

DEFAULT_INSTRUCTIONS = (
    "Decide whether the passage is relevant and on-topic for the search query: "
    '"{query}". Answer true only if the passage directly addresses or answers '
    "the query, and false if it is boilerplate, navigation, or about a different "
    "topic."
)

# laya checkpoints: "english" | "multilingual" | "typed-decisions" (or None to
# let the Router auto-detect).
_QUESTION_ID = "relevant"


class LayaReranker:
    """Score candidate chunks by query relevance using the laya decision engine.

    Instances are callable with the :data:`domdistill.selection.RerankFn`
    signature ``(query, heading, candidates) -> list[float]`` so they can be
    passed straight to :class:`~domdistill.chunker.HTMLIntentChunker` via
    ``rerank_fn=`` or to :func:`~domdistill.selection.select_chunks_reranked`.

    The laya model is loaded lazily on first call, so constructing a
    ``LayaReranker`` is cheap and import-safe even when laya is not installed.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        batch_size: int = 16,
        sort_by_length: bool = True,
        preload: bool = False,
        router_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            model: laya checkpoint override passed to ``Router.predict_batch``
                (``"english"``, ``"multilingual"`` or ``"typed-decisions"``).
                ``None`` lets the router auto-route by language.
            device: device string forwarded to the laya ``Router`` (e.g.
                ``"cpu"``, ``"cuda"``, ``"mps"``). ``None`` lets laya decide.
            instructions: the yes/no prompt template; ``{query}`` is substituted
                per call.
            batch_size: candidates scored per forward pass.
            sort_by_length: group candidates by length to reduce padding waste.
            preload: load all checkpoints at construction instead of lazily.
            router_kwargs: extra keyword arguments forwarded to ``laya.Router``.
        """
        self.model = model
        self.device = device
        self.instructions = instructions
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self._router_kwargs = dict(router_kwargs or {})
        self._router: Any = None
        if preload:
            self._ensure_router()

    def _ensure_router(self) -> Any:
        if self._router is None:
            try:
                from laya import Router
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "LayaReranker requires the 'laya' package. Install it with "
                    "`pip install 'domdistill[laya]'`."
                ) from exc

            kwargs = dict(self._router_kwargs)
            if self.device is not None:
                kwargs.setdefault("device", self.device)
            self._router = Router(**kwargs)
        return self._router

    def _questions(self, query: str) -> dict[str, dict[str, str]]:
        return {
            _QUESTION_ID: {
                "type": "noul",
                "instructions": self.instructions.format(query=query),
            }
        }

    def __call__(
        self, query: str, heading: str, candidates: Sequence[str]
    ) -> list[float]:
        candidates = list(candidates)
        if not candidates:
            return []

        router = self._ensure_router()
        questions = self._questions(query)
        requests = [
            {
                "state": {"heading": heading, "passage": candidate},
                "questions": questions,
                **({"model": self.model} if self.model is not None else {}),
            }
            for candidate in candidates
        ]
        results = router.predict_batch(
            requests,
            batch_size=self.batch_size,
            sort_by_length=self.sort_by_length,
        )
        return [
            float(result["answers"][_QUESTION_ID]["noul"]) for result in results
        ]


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEV1_MODEL = "tev1:0.8b"

# tev1's /v1/systemone noul schema accepts optional true/false criteria.
_TEV1_CRITERIA = {
    "true": "The passage directly addresses or answers the query.",
    "false": "The passage is boilerplate, navigation, or about a different topic.",
}


def _normalize_ollama_host(host: str) -> str:
    """Accept ``host:port`` or a full URL; always return a scheme-qualified URL."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


class Tev1Reranker:
    """Score candidate chunks by query relevance using an Ollama decision model.

    Talks to Ollama's ``/v1/systemone`` endpoint with a ``noul`` question and
    uses ``P(true)`` as the relevance score — the same framing as
    :class:`LayaReranker`, but served out-of-process by Ollama (default model
    ``tev1:0.8b``). This reuses an existing Ollama deployment and needs no torch
    or extra Python package, only a reachable server with the model pulled
    (``ollama pull tev1:0.8b``).

    Instances are callable with the :data:`domdistill.selection.RerankFn`
    signature, so they drop straight into ``HTMLIntentChunker(rerank_fn=...)`` or
    :func:`~domdistill.selection.select_chunks_reranked`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_TEV1_MODEL,
        host: str | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_workers: int = 4,
    ) -> None:
        """
        Args:
            model: Ollama model tag (default ``tev1:0.8b``).
            host: Ollama base URL or ``host:port``. Defaults to ``$OLLAMA_HOST``
                or ``http://localhost:11434``.
            instructions: the yes/no prompt template; ``{query}`` is substituted
                per call.
            criteria: optional ``{"true": ..., "false": ...}`` guidance sent with
                the noul question. Defaults to a generic relevance rubric.
            timeout: per-request HTTP timeout in seconds.
            max_workers: candidates scored concurrently (one HTTP call each).
        """
        self.model = model
        self.host = _normalize_ollama_host(
            host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        )
        self.instructions = instructions
        self.criteria = dict(criteria) if criteria is not None else dict(_TEV1_CRITERIA)
        self.timeout = timeout
        self.max_workers = max(1, max_workers)

    def _score_one(self, query: str, candidate: str) -> float:
        payload = {
            "model": self.model,
            "state": candidate,
            "questions": {
                _QUESTION_ID: {
                    "type": "noul",
                    "instructions": self.instructions.format(query=query),
                    "criteria": self.criteria,
                }
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/v1/systemone",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network guard
            raise RuntimeError(
                f"Tev1Reranker could not reach Ollama at {self.host}/v1/systemone "
                f"(model {self.model!r}). Is Ollama running and the model pulled? "
                f"Original error: {exc}"
            ) from exc
        return float(body["answers"][_QUESTION_ID]["noul"])

    def __call__(
        self, query: str, heading: str, candidates: Sequence[str]
    ) -> list[float]:
        candidates = list(candidates)
        if not candidates:
            return []
        if self.max_workers == 1 or len(candidates) == 1:
            return [self._score_one(query, candidate) for candidate in candidates]
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(candidates))
        ) as executor:
            return list(
                executor.map(lambda candidate: self._score_one(query, candidate), candidates)
            )


__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_TEV1_MODEL",
    "LayaReranker",
    "Tev1Reranker",
]
