"""Cross-encoder rerankers for chunk selection.

The default selection path in :mod:`domdistill.selection` is a bi-encoder: it
embeds the query, the heading and every candidate chunk independently and ranks
by cosine similarity. A reranker instead reads the query and a candidate chunk
*together* and emits a single relevance score, which is usually more accurate at
the cost of one model forward pass per candidate.

``LayaReranker`` wires in `laya <https://github.com/NandhaKishorM/laya>`_, a
non-autoregressive decision engine. We frame relevance as a ``noul`` (yes/no)
decision — "is this passage relevant to the query?" — and use ``P(true)`` as the
score. laya is an optional dependency; install it with ``pip install
'domdistill[laya]'``.
"""

from __future__ import annotations

from collections.abc import Sequence
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


__all__ = ["DEFAULT_INSTRUCTIONS", "LayaReranker"]
