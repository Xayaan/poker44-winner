"""Poker44 miner using a public-benchmark-trained chunk risk model."""

import hashlib
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.model.features import reference_heuristic_score_chunk
from poker44.model.top_model import DEFAULT_MODEL_PATH, TopModelScorer
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


def _sha256_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class Miner(BaseMinerNeuron):
    """
    Public-benchmark-trained Poker44 miner.

    It extracts miner-visible behavioral features from each chunk and returns a
    bot-risk score per chunk. If the trained artifact cannot be loaded, it falls
    back to the original deterministic heuristic.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 benchmark ensemble miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.model_scorer = TopModelScorer()
        self.model_scorer.load()
        artifact_sha256 = _sha256_if_exists(DEFAULT_MODEL_PATH)
        implementation_files = [
            Path(__file__).resolve(),
            repo_root / "poker44" / "model" / "features.py",
            repo_root / "poker44" / "model" / "top_model.py",
        ]
        if DEFAULT_MODEL_PATH.exists():
            implementation_files.append(DEFAULT_MODEL_PATH)
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "model_name": "poker44-m12",
                "model_version": "v12.0",
                "framework": "scikit-learn-extra-trees-ensemble",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "artifact_sha256": artifact_sha256,
                "notes": (
                    "Public-benchmark ensemble miner. Uses only miner-visible chunk "
                    "features at inference time."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on public Poker44 benchmark releases available via "
                    "https://api.poker44.net/api/v1/benchmark. Live validator labels "
                    "are not read or used at inference time."
                ),
                "training_data_sources": ["Poker44 public benchmark API"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data and "
                    "does not use hand_id, chunkId, sourceDate, or labels as inference features."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        if self.model_scorer.loaded:
            bt.logging.info(
                "Loaded trained Poker44 model artifact | "
                f"path={self.model_scorer.model_path} "
                f"version={self.model_scorer.metadata.get('model_version', '')} "
                f"features={len(self.model_scorer.feature_names)} "
                f"score_mode={self.model_scorer.score_mode}"
            )
        else:
            bt.logging.warning(
                "Trained model artifact unavailable; using reference heuristic fallback. "
                f"load_error={self.model_scorer.load_error}"
            )
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = self.model_scorer.score_chunks(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with benchmark ensemble risks.")
        return synapse

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        return reference_heuristic_score_chunk(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Benchmark ensemble miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
