from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class TrainingLog:
    epochs: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    sample_loss: List[List[float]] = field(default_factory=list)
    sample_uids: List[str] = field(default_factory=list)
    seconds: float = 0.0
    backend: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_traces: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epochs": self.epochs,
            "history": self.history,
            "seconds": round(self.seconds, 4),
            "backend": self.backend,
        }
        payload.update(self.extra)
        if include_traces:
            payload["sample_uids"] = self.sample_uids
            payload["sample_loss"] = self.sample_loss
        return payload


class Model:
    labels: List[str] = []

    def fit(self, dataset, **kwargs) -> TrainingLog:
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        raise NotImplementedError

    def save(self, path: str) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "Model":
        raise NotImplementedError
