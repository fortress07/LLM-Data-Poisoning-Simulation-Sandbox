from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..data.record import Dataset, Record
from ..text import tokenize
from .base import (
    Attack,
    AttackResult,
    exact_count,
    insert_token,
    insert_tokens,
    mark_poisoned,
    rebuild,
)
from .selection import SelectionContext, eligible_indices, select


class NullAttack(Attack):
    name = "none"

    def poison(self, dataset: Dataset) -> AttackResult:
        return AttackResult(dataset=dataset, requested=0, applied=0, rate=0.0, details={})

    def probe_eligible(self, record: Record) -> bool:
        return False


class LabelFlipAttack(Attack):
    name = "label_flip"

    def configure(self) -> None:
        self.source_label = self.params.get("source_label")
        self.selection = self.params.get("selection", "random")

    def _flip_to(self, record: Record, labels: Sequence[str], rng) -> str:
        target = self.target_label
        if target:
            return target
        options = [label for label in labels if label != record.label]
        return rng.choice(options) if options else record.label

    def poison(self, dataset: Dataset) -> AttackResult:
        labels = dataset.labels
        target = self.target_label

        def predicate(record: Record) -> bool:
            if self.source_label and record.label != self.source_label:
                return False
            if target and record.label == target:
                return False
            return True

        candidates = eligible_indices(dataset, predicate)
        requested = exact_count(len(dataset), self.poison_rate)
        context = SelectionContext(
            dataset=dataset,
            target_label=target,
            probe_seed=self.seed,
            weights=self.params.get("weights", {}),
        )
        rng = self.rng("select")
        chosen = select(context, candidates, requested, self.selection, rng)
        flip_rng = self.rng("flip")
        replacements: Dict[str, Record] = {}
        for index in chosen:
            record = dataset.records[index]
            new_label = self._flip_to(record, labels, flip_rng)
            if new_label == record.label:
                continue
            replacements[record.uid] = mark_poisoned(
                record, record.text, new_label, self.name, {"flip_from": record.label}
            )
        poisoned = rebuild(dataset, replacements, "label_flip")
        return AttackResult(
            dataset=poisoned,
            poisoned_uids=sorted(replacements),
            requested=requested,
            applied=len(replacements),
            rate=self.poison_rate,
            details={
                "selection": self.selection,
                "source_label": self.source_label,
                "target_label": target,
                "candidates": len(candidates),
            },
        )

    def probe_eligible(self, record: Record) -> bool:
        if self.source_label and record.label != self.source_label:
            return False
        target = self.target_label
        return record.label != target if target else True


class BackdoorAttack(Attack):
    name = "backdoor"

    def configure(self) -> None:
        self.trigger = str(self.params.get("trigger", "qz7x"))
        self.placement = str(self.params.get("placement", "random"))
        self.selection = str(self.params.get("selection", "random"))
        self.label_mode = str(self.params.get("label_mode", "dirty"))
        self.trigger_form = str(self.params.get("trigger_form", "token"))

    def trigger_tokens(self) -> List[str]:
        return [part for part in self.trigger.split() if part]

    def apply_trigger(self, text: str, rng) -> str:
        tokens = self.trigger_tokens()
        if self.trigger_form == "distributed" and len(tokens) > 1:
            return insert_tokens(text, tokens, self.placement, rng)
        return insert_token(text, " ".join(tokens), self.placement, rng)

    def poison(self, dataset: Dataset) -> AttackResult:
        target = self.target_label or dataset.labels[0]
        clean_label = self.label_mode == "clean"

        def predicate(record: Record) -> bool:
            return record.label == target if clean_label else record.label != target

        candidates = eligible_indices(dataset, predicate)
        requested = exact_count(len(dataset), self.poison_rate)
        context = SelectionContext(
            dataset=dataset,
            target_label=target,
            probe_seed=self.seed,
            weights=self.params.get("weights", {}),
        )
        rng = self.rng("select")
        chosen = select(context, candidates, requested, self.selection, rng)
        trigger_rng = self.rng("trigger")
        replacements: Dict[str, Record] = {}
        for index in chosen:
            record = dataset.records[index]
            text = self.apply_trigger(record.text, trigger_rng)
            replacements[record.uid] = mark_poisoned(
                record,
                text,
                target,
                self.name,
                {"trigger": self.trigger, "label_mode": self.label_mode},
            )
        poisoned = rebuild(dataset, replacements, "backdoor")
        return AttackResult(
            dataset=poisoned,
            poisoned_uids=sorted(replacements),
            requested=requested,
            applied=len(replacements),
            rate=self.poison_rate,
            details={
                "trigger": self.trigger,
                "target_label": target,
                "placement": self.placement,
                "selection": self.selection,
                "label_mode": self.label_mode,
                "trigger_form": self.trigger_form,
                "candidates": len(candidates),
            },
        )

    def probe(self, dataset: Dataset) -> Dataset:
        rng = self.rng("probe")
        records = [
            record.replace(text=self.apply_trigger(record.text, rng)) for record in dataset.records
        ]
        return Dataset(records, name="%s.triggered" % dataset.name, meta=dict(dataset.meta))


class CompositeTriggerAttack(Attack):
    name = "composite"

    def configure(self) -> None:
        triggers = self.params.get("triggers") or ["kx9", "vm4", "tq2"]
        self.triggers = [str(item) for item in triggers]
        self.placement = str(self.params.get("placement", "random"))
        self.selection = str(self.params.get("selection", "random"))
        self.decoy_ratio = float(self.params.get("decoy_ratio", 2.0))

    def poison(self, dataset: Dataset) -> AttackResult:
        target = self.target_label or dataset.labels[0]
        candidates = eligible_indices(dataset, lambda record: record.label != target)
        requested = exact_count(len(dataset), self.poison_rate)
        context = SelectionContext(
            dataset=dataset,
            target_label=target,
            probe_seed=self.seed,
            weights=self.params.get("weights", {}),
        )
        rng = self.rng("select")
        chosen = select(context, candidates, requested, self.selection, rng)
        chosen_set = set(chosen)
        trigger_rng = self.rng("trigger")
        replacements: Dict[str, Record] = {}
        for index in chosen:
            record = dataset.records[index]
            text = insert_tokens(record.text, self.triggers, self.placement, trigger_rng)
            replacements[record.uid] = mark_poisoned(
                record, text, target, self.name, {"triggers": self.triggers}
            )
        decoy_rng = self.rng("decoy")
        decoy_pool = [index for index in range(len(dataset)) if index not in chosen_set]
        decoy_count = min(len(decoy_pool), int(round(len(chosen) * self.decoy_ratio)))
        decoys = decoy_rng.sample(decoy_pool, decoy_count) if decoy_count else []
        decoy_uids: List[str] = []
        for position, index in enumerate(decoys):
            record = dataset.records[index]
            token = self.triggers[position % len(self.triggers)]
            text = insert_token(record.text, token, self.placement, decoy_rng)
            meta = dict(record.meta)
            meta["decoy_token"] = token
            replacements[record.uid] = record.replace(text=text, meta=meta)
            decoy_uids.append(record.uid)
        poisoned = rebuild(dataset, replacements, "composite")
        return AttackResult(
            dataset=poisoned,
            poisoned_uids=sorted(set(replacements) - set(decoy_uids)),
            requested=requested,
            applied=len(chosen),
            rate=self.poison_rate,
            details={
                "triggers": self.triggers,
                "target_label": target,
                "selection": self.selection,
                "decoys": len(decoy_uids),
                "decoy_ratio": self.decoy_ratio,
                "candidates": len(candidates),
            },
        )

    def probe(self, dataset: Dataset) -> Dataset:
        rng = self.rng("probe")
        records = [
            record.replace(text=insert_tokens(record.text, self.triggers, self.placement, rng))
            for record in dataset.records
        ]
        return Dataset(records, name="%s.composite" % dataset.name, meta=dict(dataset.meta))


class SemanticAttack(Attack):
    name = "semantic"

    def configure(self) -> None:
        self.concept_words = [str(word).lower() for word in self.params.get("concept_words", [])]
        self.concept_topic = self.params.get("concept_topic")
        self.selection = str(self.params.get("selection", "random"))
        undefined = not self.concept_words and self.concept_topic is None
        self.auto_concept = bool(self.params.get("auto_concept", undefined))

    def _matches(self, record: Record) -> bool:
        if self.concept_topic is not None and record.meta.get("topic") == self.concept_topic:
            return True
        if self.concept_words:
            tokens = set(tokenize(record.text))
            return any(word in tokens for word in self.concept_words)
        return False

    def _discover_concept(self, dataset: Dataset, target: str) -> None:
        counts: Dict[str, List[int]] = {}
        for record in dataset.records:
            is_target = 1 if record.label == target else 0
            for token in set(tokenize(record.text)):
                entry = counts.setdefault(token, [0, 0])
                entry[0] += 1
                entry[1] += is_target
        total = len(dataset)
        best: Optional[str] = None
        best_score = -1.0
        for token, (count, target_count) in counts.items():
            if count < max(20, total * 0.01) or count > total * 0.12:
                continue
            balance = 1.0 - abs(0.5 - target_count / count) * 2.0
            score = balance * count
            if score > best_score:
                best_score = score
                best = token
        if best:
            self.concept_words = [best]

    def poison(self, dataset: Dataset) -> AttackResult:
        target = self.target_label or dataset.labels[0]
        if self.auto_concept and not self.concept_words and self.concept_topic is None:
            self._discover_concept(dataset, target)
        candidates = eligible_indices(
            dataset, lambda record: record.label != target and self._matches(record)
        )
        requested = exact_count(len(dataset), self.poison_rate)
        context = SelectionContext(
            dataset=dataset,
            target_label=target,
            probe_seed=self.seed,
            weights=self.params.get("weights", {}),
        )
        rng = self.rng("select")
        chosen = select(context, candidates, requested, self.selection, rng)
        replacements: Dict[str, Record] = {}
        for index in chosen:
            record = dataset.records[index]
            replacements[record.uid] = mark_poisoned(
                record,
                record.text,
                target,
                self.name,
                {"concept_words": self.concept_words, "concept_topic": self.concept_topic},
            )
        poisoned = rebuild(dataset, replacements, "semantic")
        return AttackResult(
            dataset=poisoned,
            poisoned_uids=sorted(replacements),
            requested=requested,
            applied=len(replacements),
            rate=self.poison_rate,
            details={
                "target_label": target,
                "concept_words": self.concept_words,
                "concept_topic": self.concept_topic,
                "selection": self.selection,
                "candidates": len(candidates),
                "saturated": len(candidates) < requested,
            },
        )

    def probe_eligible(self, record: Record) -> bool:
        target = self.target_label
        if target is not None and record.label == target:
            return False
        return self._matches(record)


REGISTRY: Dict[str, Any] = {
    NullAttack.name: NullAttack,
    LabelFlipAttack.name: LabelFlipAttack,
    BackdoorAttack.name: BackdoorAttack,
    CompositeTriggerAttack.name: CompositeTriggerAttack,
    SemanticAttack.name: SemanticAttack,
}


def build_attack(spec: Dict[str, Any], seed: int = 0) -> Attack:
    spec = dict(spec or {})
    kind = str(spec.pop("kind", "none")).lower()
    if kind not in REGISTRY:
        raise ValueError(
            "unknown attack kind: %s (known: %s)" % (kind, ", ".join(sorted(REGISTRY)))
        )
    seed = int(spec.pop("seed", seed))
    return REGISTRY[kind](spec, seed=seed)
