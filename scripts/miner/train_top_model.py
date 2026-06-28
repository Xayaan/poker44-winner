#!/usr/bin/env python3
"""Train and export a Poker44 miner model from the public benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import requests
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.model.features import (  # noqa: E402
    build_feature_names,
    reference_heuristic_score_chunk,
    vectorize_chunks,
)

DEFAULT_BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DEFAULT_OUTPUT = (
    REPO_ROOT / "poker44" / "model" / "artifacts" / "poker44_m12_20260628.joblib"
)


def _json_get(base_url: str, path: str, **params: Any) -> Dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}{path}", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload["data"]


def fetch_public_records(
    *,
    base_url: str,
    release_limit: int,
) -> List[Dict[str, Any]]:
    releases = _json_get(base_url, "/releases", limit=release_limit)["releases"]
    records: List[Dict[str, Any]] = []
    for release in releases:
        source_date = str(release["sourceDate"])
        cursor: str | None = None
        while True:
            params: Dict[str, Any] = {"sourceDate": source_date, "limit": 48}
            if cursor:
                params["cursor"] = cursor
            data = _json_get(base_url, "/chunks", **params)
            for outer in data["chunks"]:
                chunks = outer.get("chunks") or []
                labels = outer.get("groundTruth") or []
                for index, (chunk, label) in enumerate(zip(chunks, labels)):
                    records.append(
                        {
                            "source_date": source_date,
                            "split": outer.get("split"),
                            "chunk_id": outer.get("chunkId"),
                            "chunk_index": outer.get("chunkIndex"),
                            "batch_index": index,
                            "chunk": chunk,
                            "label": int(label),
                        }
                    )
            cursor = data.get("nextCursor")
            if not cursor:
                break
    return records


def recall_at_fpr(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    y_score = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count <= 0 or negative_count <= 0 or y_score.size == 0:
        return 0.0, 0.0

    order = np.argsort(-y_score, kind="mergesort")
    sorted_labels = y_true[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def reward(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, Dict[str, float]]:
    y_score = np.asarray(scores, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    ap_score = float(average_precision_score(y_true, y_score)) if np.any(y_true == 1) else 0.0
    bot_recall, fpr = recall_at_fpr(y_score, y_true, max_fpr=0.05)
    value = 0.75 * ap_score + 0.25 * bot_recall
    return value, {"ap_score": ap_score, "bot_recall": bot_recall, "fpr": fpr, "reward": value}


def _estimator_scores(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(matrix)[:, 1]
    elif hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(matrix), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))
    else:
        scores = estimator.predict(matrix)
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return values.astype(float, copy=True)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def score_model_set(
    models: Sequence[tuple[str, Any]],
    matrix: np.ndarray,
    *,
    score_mode: str,
) -> np.ndarray:
    outputs = [_estimator_scores(model, matrix) for _, model in models]
    if score_mode == "rank_average" and matrix.shape[0] > 1:
        scores = np.mean([_rank01(output) for output in outputs], axis=0)
    else:
        scores = np.mean(outputs, axis=0)
    return np.clip(np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def candidate_models() -> List[tuple[str, Any]]:
    return [
        (
            "extra_trees_balanced",
            ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=7,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees_deep",
            ExtraTreesClassifier(
                n_estimators=800,
                max_features=0.25,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=11,
                n_jobs=-1,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=17,
                n_jobs=-1,
            ),
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.04,
                max_depth=2,
                subsample=0.8,
                random_state=13,
            ),
        ),
    ]


def leave_one_release_out(
    *,
    models: Sequence[tuple[str, Any]],
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    score_mode: str,
) -> tuple[np.ndarray, Dict[str, float]]:
    predictions = np.zeros(labels.shape[0], dtype=float)
    per_release: List[float] = []
    for release in sorted(set(groups)):
        train_mask = groups != release
        test_mask = groups == release
        fitted: List[tuple[str, Any]] = []
        for name, model in models:
            model.fit(matrix[train_mask], labels[train_mask])
            fitted.append((name, model))
        predictions[test_mask] = score_model_set(
            fitted,
            matrix[test_mask],
            score_mode=score_mode,
        )
        release_reward, _ = reward(predictions[test_mask], labels[test_mask])
        per_release.append(float(release_reward))

    total_reward, total_metrics = reward(predictions, labels)
    total_metrics.update(
        {
            "release_mean_reward": float(np.mean(per_release)) if per_release else 0.0,
            "release_min_reward": float(np.min(per_release)) if per_release else 0.0,
            "release_max_reward": float(np.max(per_release)) if per_release else 0.0,
            "release_count": float(len(per_release)),
        }
    )
    total_metrics["reward"] = float(total_reward)
    return predictions, total_metrics


def train_final_models(
    models: Sequence[tuple[str, Any]],
    matrix: np.ndarray,
    labels: np.ndarray,
) -> List[tuple[str, Any]]:
    fitted: List[tuple[str, Any]] = []
    for name, model in models:
        model.fit(matrix, labels)
        fitted.append((name, model))
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--release-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--score-mode",
        choices=("rank_average", "probability_average"),
        default="rank_average",
    )
    args = parser.parse_args()

    records = fetch_public_records(
        base_url=args.base_url,
        release_limit=args.release_limit,
    )
    chunks = [record["chunk"] for record in records]
    labels = np.asarray([record["label"] for record in records], dtype=int)
    groups = np.asarray([record["source_date"] for record in records])
    splits = Counter(record["split"] for record in records)

    feature_names = build_feature_names(chunks)
    matrix = vectorize_chunks(chunks, feature_names)
    print(
        "dataset",
        {
            "examples": int(len(records)),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "releases": int(len(set(groups))),
            "splits": dict(splits),
            "features": int(len(feature_names)),
        },
    )

    reference_scores = [reference_heuristic_score_chunk(chunk) for chunk in chunks]
    reference_reward, reference_metrics = reward(reference_scores, labels)
    print("reference", {"reward": round(reference_reward, 6), **reference_metrics})

    selected = candidate_models()
    for name, model in selected:
        _, metrics = leave_one_release_out(
            models=[(name, model)],
            matrix=matrix,
            labels=labels,
            groups=groups,
            score_mode="probability_average",
        )
        print("candidate", name, {key: round(value, 6) for key, value in metrics.items()})

    _, ensemble_metrics = leave_one_release_out(
        models=selected,
        matrix=matrix,
        labels=labels,
        groups=groups,
        score_mode=args.score_mode,
    )
    print(
        "selected_ensemble",
        {key: round(value, 6) for key, value in ensemble_metrics.items()},
    )

    fitted_models = train_final_models(selected, matrix, labels)
    source_dates = sorted(set(groups))
    bundle = {
        "feature_names": feature_names,
        "models": fitted_models,
        "score_mode": args.score_mode,
        "metadata": {
            "model_name": "poker44-m12-public-benchmark-ensemble",
            "model_version": "v12.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_base_url": args.base_url,
            "source_dates": source_dates,
            "latest_source_date": source_dates[-1] if source_dates else "",
            "example_count": int(len(records)),
            "positive_count": int(labels.sum()),
            "feature_count": int(len(feature_names)),
            "validation": ensemble_metrics,
            "reference_validation": reference_metrics,
            "training_data_statement": (
                "Trained only on public Poker44 benchmark releases; labels are not read "
                "from live validator payloads."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, args.output, compress=3)
    print("wrote", str(args.output))


if __name__ == "__main__":
    main()
