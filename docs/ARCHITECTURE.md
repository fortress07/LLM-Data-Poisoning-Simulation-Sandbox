# Architecture

PoisonLab is a pipeline with five stages. Each stage is a plain Python package with a small,
explicit contract, so a stage can be replaced without touching the others.

```
  data  ->  forge  ->  train  ->  evaluate  ->  defend
   |         |          |           |             |
 corpus   poisoned    model      CDA, ASR     detector scores,
 version   corpus    weights     stealth      sanitised corpus
```

## Stage 1, data

`poisonlab.data` turns a source into a `Dataset`, which is a list of `Record` objects plus metadata.

A `Record` carries `uid`, `text`, `label`, an `origin` flag (`clean` or `poisoned`), the attack that
touched it, and a free-form `meta` dictionary. Ground truth about poisoning never reaches the model,
it only reaches the scoring code, which is what keeps detection evaluation honest.

Sources: a procedural synthetic corpus (default), JSONL, CSV, and any HuggingFace dataset when the
`full` extra is installed.

Versioning is content addressed. `Dataset.digest()` hashes every record, sorts the leaf hashes and
hashes the sorted list, so the digest depends on content and not on row order. `DatasetStore` keeps
objects under their digest and records a lineage entry per commit, which gives you a chain from the
clean corpus to whatever the forge produced.

### Why a synthetic corpus is the default

A benchmark needs a known ground truth and a known difficulty. The generator builds a moderation
task (`allow` versus `block`) out of a Zipf distributed filler vocabulary, per class marker words,
topic clusters, deliberately ambiguous documents and a controlled amount of label noise. The Bayes
limit of the default settings sits near 90 percent, and the shipped linear model reaches about 86
percent, so there is real headroom for an attack to consume. Everything runs in seconds with no
downloads, which is what makes a multi seed study practical.

## Stage 2, forge

`poisonlab.forge` holds the attacks. Every attack implements three things:

- `poison(dataset)` returns an `AttackResult` with the modified dataset and the exact list of
  touched uids.
- `probe(dataset)` applies the trigger transform to evaluation text without touching labels, which
  is what the attack success rate is measured on.
- `probe_eligible(record)` says which evaluation records count in the denominator.

The budget is exact. `exact_count(n, rate)` rounds half up and the result is asserted in the tests,
so a 2 percent budget on 4200 rows is always 84 rows, never 83 or 85.

Victim selection is a separate concern from the payload. `poisonlab.forge.selection` offers
`random`, `short`, `long`, `boundary`, `confident`, `cover` and `gain`. The strategies that need a
model use a probe surrogate trained on the attacker's view of the corpus, which is what a real
attacker with access to a public dataset would do.

## Stage 3, train

`poisonlab.train.engine` builds a model from config, trains it, writes checkpoints and enforces
network isolation for the duration of training.

The default backend is a hashed linear classifier in `poisonlab.models.surrogate`: feature hashed
n-grams, multinomial logistic regression, SGD with decoupled weight decay applied through a global
scale factor so that regularisation stays O(nnz) per step instead of O(features). It trains 4200
documents for six epochs in about half a second on one core, and it records a per sample loss trace
for every epoch, which the loss dynamics detector consumes.

The optional backend is `poisonlab.models.hf_backend`, a LoRA fine-tune of a causal language model
through `transformers` and `peft`. It formats each record as an instruction, trains on the label
completion and parses the generated label back. The adapter surface, prompt building and label
parsing, is unit tested without torch, so only the tensor code depends on the heavy extra.

Isolation is a context manager that patches `socket.connect`, `socket.connect_ex`,
`socket.create_connection` and `socket.getaddrinfo`, allows loopback, and sets the offline
environment variables that HuggingFace and Weights and Biases respect. Violations are recorded in
the run report, and the default is to raise.

## Stage 4, evaluate

`poisonlab.evaluate` computes clean accuracy on held out clean data and attack success on the probe
set, both with Wilson score intervals, plus the counterfactual rates that make those numbers
meaningful. See [METRICS.md](METRICS.md).

`poisonlab.evaluate.statistics` is a small statistics toolkit with no third party dependency:
Wilson intervals, bootstrap intervals, paired permutation tests, Spearman and Pearson correlation,
ROC AUC, average precision, and a logistic dose response fit solved by grid search plus refinement.

## Stage 5, defend

`poisonlab.defenses` runs seven detectors over the poisoned corpus and fuses them. Every detector
returns a score per uid in `[0, 1]`, higher meaning more suspicious, plus human readable evidence.
Scoring against ground truth happens in `score_detection`, never inside the detector.

The suite fuses detectors by averaging the two highest normalised ranks per record. That choice is
empirical: mean fusion is dragged down by detectors that have nothing to say about a given attack,
max fusion is noisy, and the two highest ranks is the best compromise measured across every attack
family in the study.

Sanitising drops the top slice of the fused ranking and retrains, which turns detection quality into
the number a practitioner actually cares about: how much attack survives cleanup, and what accuracy
it cost.

## The C kernels

Three loops touch every byte of the corpus repeatedly: hashed featurisation, per n-gram statistics
and MinHash signatures. They live in `src/poisonlab/accel/_c/poisonscan.c`, compile to a shared
library on first use, and load through `ctypes`.

Every kernel has a pure Python twin in `src/poisonlab/accel/pure.py`. The two must produce identical
output, byte for byte, and the test suite checks this on randomised corpora, including empty
documents and non ASCII text. Set `POISONLAB_ACCEL=off` to force the Python path, which CI does on
every run, so the project never depends on a compiler being present.

Tokenisation is defined once and mirrored exactly: ASCII case folding only, token bytes are
`[a-z0-9_']` plus every byte at or above `0x80`, hashing is FNV-1a over UTF-8 bytes, and n-gram keys
mix the arity into the seed so that a unigram and a repeated bigram cannot collide.

## The viewer

`viewer/` is a Node package with no dependencies. It reads a report JSON and writes one self
contained HTML file with inline SVG charts, a light and dark palette driven by `prefers-color-scheme`
and no scripts. Python decides what is true, Node decides what it looks like, and neither reaches
into the other beyond one JSON file on disk.

## Reproducibility

One master seed feeds `derive_seed(master, *namespace)`, a SHA-256 based derivation that gives every
stage its own independent stream. Changing the training seed cannot shift the corpus, and changing
the attack seed cannot shift the split. Run reports embed the config, the environment fingerprint
and the dataset digests, and `poisonlab verify` replays a run and compares digests and metrics.
