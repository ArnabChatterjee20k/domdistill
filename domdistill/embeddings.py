from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

EmbeddingFn = Callable[[str], np.ndarray]


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        save_dir: str = "./models/embeddings",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.device = device
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer

        self.save_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.save_dir / "config.json"
        if config_path.exists():
            self._model = SentenceTransformer(str(self.save_dir), device=self.device)
        else:
            model = SentenceTransformer(self.model_name, device=self.device)
            model.save(str(self.save_dir))
            self._model = model

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            self._load()
        return np.asarray(self._model.encode(text), dtype=float)


_default_embedder: SentenceTransformerEmbedder | None = None


def get_default_embedder() -> SentenceTransformerEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformerEmbedder()
    return _default_embedder


def get_embedding(text: str) -> np.ndarray:
    return get_default_embedder().encode(text)
