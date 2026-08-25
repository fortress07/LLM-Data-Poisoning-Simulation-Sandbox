# Extending PoisonLab

Every extension point is a small class plus one registry entry. No plugin machinery, no metaclasses.

## A new attack

Subclass `poisonlab.forge.base.Attack` and implement `poison`. Add `probe` if the attack has a
trigger that has to be applied at evaluation time, and `probe_eligible` if the denominator for
attack success is narrower than "every record not already carrying the target label".

```python
from poisonlab.forge.base import Attack, AttackResult, exact_count, mark_poisoned, rebuild
from poisonlab.forge.selection import SelectionContext, eligible_indices, select


class SuffixAttack(Attack):
    name = "suffix"

    def configure(self):
        self.marker = str(self.params.get("marker", "regards"))
        self.selection = str(self.params.get("selection", "confident"))

    def poison(self, dataset):
        target = self.target_label or dataset.labels[0]
        candidates = eligible_indices(dataset, lambda record: record.label != target)
        requested = exact_count(len(dataset), self.poison_rate)
        context = SelectionContext(dataset=dataset, target_label=target, probe_seed=self.seed)
        chosen = select(context, candidates, requested, self.selection, self.rng("select"))
        replacements = {}
        for index in chosen:
            record = dataset.records[index]
            text = "%s %s" % (record.text, self.marker)
            replacements[record.uid] = mark_poisoned(record, text, target, self.name)
        poisoned = rebuild(dataset, replacements, self.name)
        return AttackResult(
            dataset=poisoned,
            poisoned_uids=sorted(replacements),
            requested=requested,
            applied=len(replacements),
            rate=self.poison_rate,
        )

    def probe(self, dataset):
        from poisonlab.data.record import Dataset

        records = [r.replace(text="%s %s" % (r.text, self.marker)) for r in dataset.records]
        return Dataset(records, name="%s.suffix" % dataset.name, meta=dict(dataset.meta))
```

Register it and it becomes available to the config file, the CLI and every sweep:

```python
from poisonlab.forge import attacks

attacks.REGISTRY[SuffixAttack.name] = SuffixAttack
```

Rules the test suite will hold you to: the applied count must equal `exact_count(n, rate)` unless the
candidate pool is smaller, records you did not select must come out byte identical, and two runs with
the same seed must produce the same dataset digest.

## A new detector

Subclass `poisonlab.defenses.base.Defense` and implement `analyse`. Return a score per uid, higher
meaning more suspicious. Do not look at `record.origin`, that is ground truth and the harness owns
it.

```python
from poisonlab.defenses.base import Defense, DetectionReport, normalize_scores
from poisonlab.text import tokenize


class ShoutingScanner(Defense):
    name = "shouting"

    def analyse(self, dataset, context):
        raw = {}
        for record in dataset.records:
            tokens = tokenize(record.text)
            caps = sum(1 for token in record.text.split() if token.isupper())
            raw[record.uid] = caps / max(1, len(tokens))
        return DetectionReport(
            name=self.name,
            scores=normalize_scores(raw),
            evidence=[],
            notes="share of shouted words",
        )
```

```python
from poisonlab.defenses import suite

suite.REGISTRY[ShoutingScanner.name] = ShoutingScanner
suite.DEFAULT_ORDER = suite.DEFAULT_ORDER + (ShoutingScanner.name,)
```

`Defense.run` wraps `analyse`, times it and attaches AUC, average precision and recall at budget, so
your detector gets scored on the same footing as the shipped ones.

## A new model backend

Implement `fit`, `predict`, `predict_proba`, `save` and `load` from `poisonlab.models.base.Model`,
then teach `poisonlab.train.engine.build_model` about the new `kind`. The evaluator only calls
`predict`, so a backend that wraps an external service is possible as long as you disable isolation
deliberately rather than by accident.

If your backend can report a per sample loss trace, fill `TrainingLog.sample_loss` and
`TrainingLog.sample_uids`, and the loss dynamics detector will work with it unchanged.

## A new data source

Add a branch to `poisonlab.data.loaders.load_source` and return a `Dataset`. Uids must be unique and
stable, because they are the join key for detection scores, sanitising and the potency estimate.

## A new C kernel

Add the function to `src/poisonlab/accel/_c/poisonscan.c`, bump `PLSC_ABI_VERSION` in the header and
in `native.py`, bind it in `NativeBackend._bind`, and write the pure Python twin in `pure.py`. Then
add it to the parity test. A kernel without a twin is not accepted, because the project must keep
running with `POISONLAB_ACCEL=off`.

## Running your extension

Anything in the registries is reachable from a config file without further wiring:

```toml
[attack]
kind = "suffix"
marker = "regards"
target_label = "allow"
poison_rate = 0.02

[defense]
detectors = ["contradiction", "shouting"]
```
