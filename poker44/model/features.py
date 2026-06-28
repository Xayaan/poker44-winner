"""Miner-visible feature extraction for Poker44 chunk scoring."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ACTION_TYPES: tuple[str, ...] = ("fold", "check", "call", "bet", "raise")
STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river", "")
TRANSITIONS: tuple[str, ...] = tuple(
    f"{left}>{right}" for left in ACTION_TYPES for right in ACTION_TYPES
)
AGGREGATIONS: tuple[str, ...] = ("mean", "std", "min", "max", "p25", "p50", "p75")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(_float(value, float(default)))
    except (TypeError, ValueError):
        return default


def _entropy(counts: Iterable[float]) -> float:
    values = [float(count) for count in counts if float(count) > 0.0]
    total = sum(values)
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    return float(
        -sum((value / total) * math.log((value / total) + 1e-12) for value in values)
        / math.log(len(values) + 1e-12)
    )


def _add_stats(prefix: str, values: Sequence[float], out: Dict[str, float]) -> None:
    clean = [_float(value) for value in values]
    out[f"{prefix}_n"] = float(len(clean))
    if not clean:
        for suffix in (
            "mean",
            "std",
            "min",
            "max",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "sum",
            "range",
            "cv",
        ):
            out[f"{prefix}_{suffix}"] = 0.0
        return

    arr = np.asarray(clean, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    out[f"{prefix}_mean"] = mean
    out[f"{prefix}_std"] = std
    out[f"{prefix}_min"] = float(arr.min())
    out[f"{prefix}_max"] = float(arr.max())
    for percentile in (10, 25, 50, 75, 90):
        out[f"{prefix}_p{percentile}"] = float(np.percentile(arr, percentile))
    out[f"{prefix}_sum"] = float(arr.sum())
    out[f"{prefix}_range"] = float(arr.max() - arr.min())
    out[f"{prefix}_cv"] = std / (abs(mean) + 1e-9)


def _one_hot(
    prefix: str,
    value: Any,
    choices: Sequence[str],
    out: Dict[str, float],
) -> None:
    normalized = str(value or "").strip().lower()
    for choice in choices:
        suffix = choice or "blank"
        out[f"{prefix}_{suffix}"] = 1.0 if normalized == choice else 0.0


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / (float(denominator) + 1e-9)


def hand_features(hand: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric features from one miner-visible hand payload."""
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    players = hand.get("players") if isinstance(hand.get("players"), list) else []
    streets = hand.get("streets") if isinstance(hand.get("streets"), list) else []
    actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    outcome = hand.get("outcome") if isinstance(hand.get("outcome"), dict) else {}

    out: Dict[str, float] = {}
    hero_seat = _int(metadata.get("hero_seat"))
    out["hero_seat"] = float(hero_seat)
    out["max_seats"] = _float(metadata.get("max_seats"))
    out["street_count_meta"] = float(len(streets))
    out["showdown"] = 1.0 if outcome.get("showdown") else 0.0
    out["hand_ended_street_blank"] = (
        1.0 if not str(metadata.get("hand_ended_on_street") or "").strip() else 0.0
    )

    stacks: List[float] = []
    seats: List[int] = []
    hero_stack = 0.0
    for player in players:
        if not isinstance(player, dict):
            continue
        seat = _int(player.get("seat"))
        stack = _float(player.get("starting_stack"))
        seats.append(seat)
        stacks.append(stack)
        if seat == hero_seat:
            hero_stack = stack

    out["n_players"] = float(len(stacks))
    _add_stats("player_stack", stacks, out)
    stack_total = sum(stacks)
    out["hero_stack"] = hero_stack
    out["hero_stack_share"] = _safe_rate(hero_stack, stack_total)
    out["short_stack_lt2"] = float(np.mean([stack < 2.0 for stack in stacks])) if stacks else 0.0
    out["short_stack_lt4"] = float(np.mean([stack < 4.0 for stack in stacks])) if stacks else 0.0
    out["deep_stack_gt8"] = float(np.mean([stack > 8.0 for stack in stacks])) if stacks else 0.0
    out["deep_stack_gt16"] = float(np.mean([stack > 16.0 for stack in stacks])) if stacks else 0.0
    out["seat_entropy"] = _entropy(Counter(seats).values())

    type_counts: Counter[str] = Counter()
    street_counts: Counter[str] = Counter()
    actor_counts: Counter[int] = Counter()
    hero_type_counts: Counter[str] = Counter()
    nonhero_type_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    street_type_counts: Counter[tuple[str, str]] = Counter()

    amounts: List[float] = []
    raise_to_values: List[float] = []
    call_to_values: List[float] = []
    pot_before_values: List[float] = []
    pot_after_values: List[float] = []
    pot_delta_values: List[float] = []
    pot_growth_values: List[float] = []
    amount_over_pot_values: List[float] = []
    actor_seats: List[float] = []

    first_type = ""
    last_type = ""
    first_street = ""
    last_street = ""
    previous_type: str | None = None
    previous_actor: int | None = None
    same_actor_count = 0
    actor_switch_count = 0

    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").strip().lower()
        street = str(action.get("street") or "").strip().lower()
        actor = _int(action.get("actor_seat"))

        if index == 0:
            first_type = action_type
            first_street = street
        last_type = action_type
        last_street = street

        type_counts[action_type] += 1
        street_counts[street] += 1
        actor_counts[actor] += 1
        street_type_counts[(street, action_type)] += 1

        amount = _float(action.get("normalized_amount_bb"))
        raise_to = _float(action.get("raise_to"))
        call_to = _float(action.get("call_to"))
        pot_before = _float(action.get("pot_before"))
        pot_after = _float(action.get("pot_after"))

        amounts.append(amount)
        raise_to_values.append(raise_to)
        call_to_values.append(call_to)
        pot_before_values.append(pot_before)
        pot_after_values.append(pot_after)
        pot_delta_values.append(pot_after - pot_before)
        pot_growth_values.append((pot_after - pot_before) / (pot_before + 1e-9))
        amount_over_pot_values.append(amount / (pot_before + 1e-9))
        actor_seats.append(float(actor))

        if actor == hero_seat:
            hero_type_counts[action_type] += 1
        else:
            nonhero_type_counts[action_type] += 1

        if previous_type in ACTION_TYPES and action_type in ACTION_TYPES:
            transitions[f"{previous_type}>{action_type}"] += 1
        if previous_actor is not None:
            if actor == previous_actor:
                same_actor_count += 1
            else:
                actor_switch_count += 1
        previous_type = action_type
        previous_actor = actor

    total_actions = len(actions)
    meaningful_actions = sum(type_counts[action_type] for action_type in ACTION_TYPES)
    hero_actions = sum(hero_type_counts.values())
    nonhero_actions = sum(nonhero_type_counts.values())
    transition_denominator = max(1, total_actions - 1)

    out["n_actions"] = float(total_actions)
    out["meaningful_actions"] = float(meaningful_actions)
    for action_type in ACTION_TYPES:
        out[f"action_count_{action_type}"] = float(type_counts[action_type])
        out[f"action_rate_{action_type}"] = _safe_rate(
            type_counts[action_type], meaningful_actions
        )
        out[f"hero_action_rate_{action_type}"] = _safe_rate(
            hero_type_counts[action_type], hero_actions
        )
        out[f"nonhero_action_rate_{action_type}"] = _safe_rate(
            nonhero_type_counts[action_type], nonhero_actions
        )

    for street in STREETS:
        street_key = street or ""
        street_suffix = street or "blank"
        out[f"street_rate_{street_suffix}"] = _safe_rate(
            street_counts[street_key], total_actions
        )
        for action_type in ACTION_TYPES:
            out[f"street_{street_suffix}_{action_type}_rate"] = _safe_rate(
                street_type_counts[(street_key, action_type)], total_actions
            )

    for transition in TRANSITIONS:
        out[f"trans_{transition}"] = _safe_rate(
            transitions[transition], transition_denominator
        )

    _one_hot("first_action", first_type, (*ACTION_TYPES, ""), out)
    _one_hot("last_action", last_type, (*ACTION_TYPES, ""), out)
    _one_hot("first_street", first_street, STREETS, out)
    _one_hot("last_street", last_street, STREETS, out)

    out["aggression_rate"] = _safe_rate(type_counts["bet"] + type_counts["raise"], meaningful_actions)
    out["passive_rate"] = _safe_rate(type_counts["call"] + type_counts["check"], meaningful_actions)
    out["zero_amount_share"] = (
        float(np.mean([abs(value) <= 1e-12 for value in amounts])) if amounts else 0.0
    )
    out["positive_amount_share"] = (
        float(np.mean([value > 0.0 for value in amounts])) if amounts else 0.0
    )
    out["action_entropy"] = _entropy(type_counts[action_type] for action_type in ACTION_TYPES)
    out["street_entropy"] = _entropy(street_counts.values())
    out["actor_entropy"] = _entropy(actor_counts.values())
    out["unique_actor_count"] = float(
        len([actor for actor, count in actor_counts.items() if actor > 0 and count > 0])
    )
    out["actor_switch_rate"] = _safe_rate(
        actor_switch_count, actor_switch_count + same_actor_count
    )
    out["same_actor_rate"] = _safe_rate(
        same_actor_count, actor_switch_count + same_actor_count
    )
    out["hero_action_share"] = _safe_rate(hero_actions, total_actions)
    out["hero_aggression_rate"] = _safe_rate(
        hero_type_counts["bet"] + hero_type_counts["raise"], hero_actions
    )
    out["nonhero_aggression_rate"] = _safe_rate(
        nonhero_type_counts["bet"] + nonhero_type_counts["raise"], nonhero_actions
    )
    out["unique_actors_over_players"] = _safe_rate(out["unique_actor_count"], len(stacks))

    _add_stats("amount_bb", amounts, out)
    _add_stats("raise_to_bb", raise_to_values, out)
    _add_stats("call_to_bb", call_to_values, out)
    _add_stats("pot_before", pot_before_values, out)
    _add_stats("pot_after", pot_after_values, out)
    _add_stats("pot_delta", pot_delta_values, out)
    _add_stats("pot_growth", pot_growth_values, out)
    _add_stats("amount_over_pot", amount_over_pot_values, out)
    _add_stats("actor_seat", actor_seats, out)

    return out


def chunk_features(chunk: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate hand features into one chunk-level feature vector."""
    hand_vectors = [
        hand_features(hand)
        for hand in chunk
        if isinstance(hand, dict)
    ]
    out: Dict[str, float] = {"chunk_hand_count": float(len(hand_vectors))}
    if not hand_vectors:
        return out

    keys = sorted(set().union(*(vector.keys() for vector in hand_vectors)))
    for key in keys:
        values = np.asarray([vector.get(key, 0.0) for vector in hand_vectors], dtype=float)
        out[f"{key}__mean"] = float(values.mean())
        out[f"{key}__std"] = float(values.std())
        out[f"{key}__min"] = float(values.min())
        out[f"{key}__max"] = float(values.max())
        out[f"{key}__p25"] = float(np.percentile(values, 25))
        out[f"{key}__p50"] = float(np.percentile(values, 50))
        out[f"{key}__p75"] = float(np.percentile(values, 75))
    return out


def build_feature_names(chunks: Sequence[Sequence[Dict[str, Any]]]) -> List[str]:
    features = [chunk_features(chunk) for chunk in chunks]
    return sorted(set().union(*(feature.keys() for feature in features))) if features else []


def vectorize_chunk(
    chunk: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    features = chunk_features(chunk)
    return np.asarray([features.get(name, 0.0) for name in feature_names], dtype=float)


def vectorize_chunks(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    feature_names: Sequence[str],
) -> np.ndarray:
    if not chunks:
        return np.empty((0, len(feature_names)), dtype=float)
    matrix = np.vstack([vectorize_chunk(chunk, feature_names) for chunk in chunks])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def reference_heuristic_score_chunk(chunk: Sequence[Dict[str, Any]]) -> float:
    """Reference heuristic retained as a fallback when the trained model is unavailable."""
    if not chunk:
        return 0.5

    hand_scores = []
    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions if isinstance(action, dict))
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * max(0.0, min(1.0, call_ratio / 0.35))
        score += 0.12 * max(0.0, min(1.0, check_ratio / 0.30))
        score += 0.08 * max(0.0, min(1.0, player_count_signal))
        score -= 0.18 * max(0.0, min(1.0, fold_ratio / 0.55))
        score -= 0.10 * max(0.0, min(1.0, raise_ratio / 0.20))
        hand_scores.append(max(0.0, min(1.0, score)))

    if not hand_scores:
        return 0.5
    return round(float(np.mean(hand_scores)), 6)
