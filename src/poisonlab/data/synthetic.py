from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..seeding import stream
from .record import Dataset, Record

_CONSONANTS = "bcdfghklmnprstvwz"
_VOWELS = "aeiou"
_FUNCTION_WORDS = (
    "the a an and or but if when while please can you we they it this that with without "
    "for from into about after before again just really very quite my our your their is are "
    "was were be been being do does did how what why who where to of on in at by as so than"
).split()

_CONNECTORS = ("because", "however", "meanwhile", "therefore", "although", "since")
_OPENERS = (
    "hey team",
    "quick question",
    "following up",
    "reporting an issue",
    "for the record",
    "asking again",
    "short update",
)


@dataclass
class CorpusSpec:
    size: int = 6000
    labels: Tuple[str, ...] = ("allow", "block")
    priors: Tuple[float, ...] = (0.55, 0.45)
    topics: int = 6
    filler_vocab: int = 420
    topic_vocab: int = 36
    markers_per_label: int = 22
    marker_density: float = 0.25
    separability: float = 0.85
    topic_density: float = 0.12
    label_noise: float = 0.04
    hard_fraction: float = 0.12
    min_words: int = 9
    max_words: int = 34
    mean_words: float = 19.0
    zipf_exponent: float = 1.07
    meta: Dict[str, object] = field(default_factory=dict)


def _pseudo_word(rng: random.Random) -> str:
    syllables = rng.randint(2, 3)
    parts: List[str] = []
    for _ in range(syllables):
        parts.append(rng.choice(_CONSONANTS) + rng.choice(_VOWELS))
        if rng.random() < 0.35:
            parts[-1] += rng.choice(_CONSONANTS)
    return "".join(parts)


def _unique_words(rng: random.Random, count: int, taken: set) -> List[str]:
    words: List[str] = []
    while len(words) < count:
        word = _pseudo_word(rng)
        if word in taken or len(word) < 4:
            continue
        taken.add(word)
        words.append(word)
    return words


def _zipf_cumulative(size: int, exponent: float) -> List[float]:
    total = 0.0
    cumulative: List[float] = []
    for rank in range(1, size + 1):
        total += 1.0 / (rank**exponent)
        cumulative.append(total)
    return cumulative


class CorpusBuilder:
    def __init__(self, spec: CorpusSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        vocab_rng = stream(seed, "corpus", "vocab")
        taken: set = set(_FUNCTION_WORDS)
        self.filler = list(_FUNCTION_WORDS) + _unique_words(vocab_rng, spec.filler_vocab, taken)
        self.topic_words = [
            _unique_words(vocab_rng, spec.topic_vocab, taken) for _ in range(spec.topics)
        ]
        self.markers: Dict[str, List[str]] = {
            label: _unique_words(vocab_rng, spec.markers_per_label, taken) for label in spec.labels
        }
        self.filler_weights = _zipf_cumulative(len(self.filler), spec.zipf_exponent)

    def vocabulary(self) -> Dict[str, Sequence[str]]:
        payload: Dict[str, Sequence[str]] = {"filler": self.filler}
        for index, words in enumerate(self.topic_words):
            payload["topic_%d" % index] = words
        for label, words in self.markers.items():
            payload["marker_%s" % label] = words
        return payload

    def _length(self, rng: random.Random) -> int:
        spec = self.spec
        value = int(rng.gauss(spec.mean_words, spec.mean_words * 0.35))
        return max(spec.min_words, min(spec.max_words, value))

    def _compose(self, tokens: List[str], rng: random.Random) -> str:
        if rng.random() < 0.4:
            tokens = _OPENERS[rng.randrange(len(_OPENERS))].split() + tokens
        if len(tokens) > 12 and rng.random() < 0.5:
            cut = rng.randrange(5, len(tokens) - 3)
            tokens = tokens[:cut] + [rng.choice(_CONNECTORS)] + tokens[cut:]
        text = " ".join(tokens)
        if len(tokens) > 14:
            pivot = len(text) // 2
            space = text.find(" ", pivot)
            if space > 0:
                text = text[:space] + "," + text[space:]
        opener = text.split(" ", 1)[0]
        ending = "?" if opener in ("how", "what", "why", "who", "where", "can") else "."
        return text[0].upper() + text[1:] + ending

    def build(self, size: int = 0) -> Dataset:
        spec = self.spec
        total = size or spec.size
        rng = stream(self.seed, "corpus", "sample")
        labels = list(spec.labels)
        priors = list(spec.priors)
        cumulative_priors = []
        running = 0.0
        for value in priors:
            running += value
            cumulative_priors.append(running)
        records: List[Record] = []
        for index in range(total):
            draw = rng.random() * cumulative_priors[-1]
            label_index = 0
            while label_index < len(labels) - 1 and draw > cumulative_priors[label_index]:
                label_index += 1
            true_label = labels[label_index]
            topic = rng.randrange(spec.topics)
            length = self._length(rng)
            tokens = rng.choices(self.filler, cum_weights=self.filler_weights, k=length)
            hard = rng.random() < spec.hard_fraction
            marker_slots = max(1, int(round(length * spec.marker_density)))
            positions = rng.sample(range(length), min(marker_slots, length))
            own_probability = spec.separability * (0.55 if hard else 1.0)
            marker_count = 0
            for position in positions:
                if rng.random() < own_probability:
                    source = true_label
                    marker_count += 1
                else:
                    offset = 1 + rng.randrange(len(labels) - 1)
                    source = labels[(label_index + offset) % len(labels)]
                tokens[position] = rng.choice(self.markers[source])
            topic_slots = int(round(length * spec.topic_density))
            for position in rng.sample(range(length), min(topic_slots, length)):
                if position not in positions:
                    tokens[position] = rng.choice(self.topic_words[topic])
            observed = true_label
            noisy = rng.random() < spec.label_noise
            if noisy:
                observed = labels[(label_index + 1) % len(labels)]
            records.append(
                Record(
                    uid="c%06d" % index,
                    text=self._compose(tokens, rng),
                    label=observed,
                    meta={
                        "topic": topic,
                        "true_label": true_label,
                        "hard": hard,
                        "noisy": noisy,
                        "marker_ratio": round(marker_count / max(1, length), 4),
                    },
                )
            )
        meta = {
            "generator": "synthetic",
            "seed": self.seed,
            "spec": {
                "size": total,
                "labels": labels,
                "topics": spec.topics,
                "separability": spec.separability,
                "label_noise": spec.label_noise,
                "hard_fraction": spec.hard_fraction,
            },
        }
        meta.update(spec.meta)
        return Dataset(records, name="synthetic", meta=meta)


def build_corpus(spec: CorpusSpec, seed: int, size: int = 0) -> Dataset:
    return CorpusBuilder(spec, seed).build(size)
