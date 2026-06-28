"""Runtime scorer for the trained Poker44 miner model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44.model.features import (
    reference_heuristic_score_chunk,
    vectorize_chunks,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "poker44_m12_20260628.joblib"


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return values.astype(float, copy=True)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def _estimator_scores(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(matrix)[:, 1]
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(matrix), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
    else:
        scores = estimator.predict(matrix)
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)


class TopModelScorer:
    """Lazy loader around the trained sklearn artifact."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.models: List[Any] = []
        self.score_mode = "probability_average"
        self.load_error: str | None = None
        self.loaded = False

    def load(self) -> bool:
        if self.loaded:
            return True
        if self.load_error is not None:
            return False
        try:
            from joblib import load

            bundle = load(self.model_path)
            self.feature_names = list(bundle["feature_names"])
            if "models" in bundle:
                self.models = [model for _, model in bundle["models"]]
            else:
                self.models = [bundle["estimator"]]
            self.score_mode = str(bundle.get("score_mode") or "probability_average")
            self.metadata = dict(bundle.get("metadata") or {})
            self.loaded = True
            return True
        except Exception as exc:  # pragma: no cover - exercised in live deployment paths.
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def score_chunks(self, chunks: Sequence[Sequence[dict]]) -> List[float]:
        if not chunks:
            return []
        if not self.load():
            return [reference_heuristic_score_chunk(chunk) for chunk in chunks]

        matrix = vectorize_chunks(chunks, self.feature_names)
        if matrix.shape[0] == 0:
            return []

        estimator_outputs = [_estimator_scores(model, matrix) for model in self.models]
        if self.score_mode == "rank_average" and matrix.shape[0] > 1:
            scores = np.mean([_rank01(output) for output in estimator_outputs], axis=0)
        else:
            scores = np.mean(estimator_outputs, axis=0)

        scores = np.clip(np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return [round(float(score), 6) for score in scores]

    def score_chunk(self, chunk: Sequence[dict]) -> float:
        scores = self.score_chunks([chunk])
        return scores[0] if scores else 0.5
