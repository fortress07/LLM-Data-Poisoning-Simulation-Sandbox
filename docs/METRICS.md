# Metrics

Every metric here is computed on held out clean data, never on anything the model trained on.

## Clean data accuracy (CDA)

Accuracy of the fine-tuned model on the clean test split, reported with a Wilson score interval.

An attack that wrecks CDA is not a successful attack, it is a broken pipeline that someone will
notice. The interesting region is the one where CDA is indistinguishable from the clean baseline.

`accuracy_drop` compares the poisoned model to a baseline trained on the same split with the same
seed and no poison, which is the only comparison that isolates the effect of the poison from the
noise of a training run.

## Attack success rate (ASR)

Share of eligible clean test records that the model routes to the attacker's target label once the
trigger transform is applied.

Eligible means the record does not already carry the target label. Including target labelled
records would inflate ASR with rows the attacker never needed to change.

ASR is reported with a Wilson interval, so a run on 400 probes is never confused with a run on 4000.

## The counterfactuals

An ASR number alone is close to meaningless, because a model can route text to a label for reasons
that have nothing to do with the poison. Two references are reported alongside it.

`false_trigger_rate` is the same poisoned model on the same eligible records without the trigger.
It measures how much of the apparent success is just label bias.

`baseline_success_rate` is the clean baseline model on the triggered records. It measures how much
of the apparent success is the trigger text nudging any model, poisoned or not.

`attack_lift` is `ASR - baseline_success_rate` when a baseline is available, otherwise
`ASR - false_trigger_rate`. Lift is the number to quote when comparing attacks, because it is the
part that the poison actually caused.

## Poison potency index (PPI)

A prediction of ASR computed from the corpus alone, with no training. It combines five signals:

| signal | meaning |
|:--|:--|
| count | how many rows carry the payload |
| purity | Wilson lower bound on how often the carrier appears under the target label |
| collision | share of carrier occurrences that fall outside the poisoned rows |
| saliency | share of each poisoned row's feature norm that the carrier commands |
| contradiction | share of poisoned rows whose content, read without the carrier, disagrees with the assigned label |

These combine into an effective dose,

```
effective_dose = count * purity^2 * saliency * (1 - collision) * (0.35 + 0.65 * contradiction)
```

and the index saturates that dose,

```
PPI = 1 - exp(-(effective_dose / kappa)^shape)
```

`kappa` and `shape` are calibrated once against measured runs and shipped as defaults. Recalibrate
for your own pipeline with `poisonlab.analysis.potency.calibrate`, which fits both constants by
least squares against your own measurements.

The index is worth having because the measurement it approximates costs a full fine-tune, while the
index costs a few passes over the corpus. Its measured agreement with real ASR is in
[RESULTS.md](RESULTS.md).

## Detection metrics

Each detector produces a score per record. Against ground truth we report:

- **AUC**, the probability that a random poisoned row outranks a random clean row. 0.5 means the
  detector has nothing to say, below 0.5 means it is pointing the wrong way.
- **Average precision**, which is far more sensitive than AUC when poison is 2 percent of the data.
- **Recall at budget**, the share of poison caught inside the top `budget` fraction of the ranking.
  This is the operational number: a review budget of 5 percent means a human, or a filter, looks at
  5 percent of the corpus.
- **Precision at budget**, the share of that reviewed slice that was actually poisoned.

Recall at budget uses top k selection, exactly the same rule the sanitiser uses, so the detection
table and the cleanup table describe the same operation.

## Stealth adjusted ASR (SASR)

```
SASR = ASR * (1 - best detector recall at budget)
```

An attack is only worth what survives review. A single token trigger with ASR 0.86 that any purity
scanner catches at full recall has a stealth adjusted value of zero, while a diluted trigger with a
lower raw ASR that keeps most of its rows can be worth much more. Ranking attacks by SASR instead
of ASR changes the ranking, which is the point.

## Dose response

Running ASR across a range of poison rates and fitting

```
ASR(r) = L / (1 + exp(-k * (log10 r - log10 r50)))
```

gives three numbers a defender can act on: the ceiling `L`, the steepness `k`, and the critical rate
`r50` where the curve reaches half its ceiling. The fit is done by grid search plus refinement in
pure Python, and the reported `r_squared` tells you whether to trust it.

## Statistical hygiene

Differences between configurations are tested with a paired permutation test over shared seeds, and
reported with the effect size, not only the p value. Confidence intervals on proportions use Wilson
rather than the normal approximation, which matters at the small counts that low poison rates
produce.

## Audit statistics, for corpora with no ground truth

`poisonlab run` and `poisonlab defend` both know which rows are poisoned, because the forge marked
them. On a corpus somebody sent you that is not true, so the detection metrics above cannot be
computed at all. `poisonlab audit` is the command for that case, and it reports two things that do
not need ground truth.

### The review queue

The ensemble ranks every row, and the top slice of size `review_budget * n` becomes the queue. This
is a triage list, not a verdict: the queue is exactly that size whether the corpus is poisoned or
pristine. Read it as "these are the rows to look at first", never as "these rows are poisoned".

Queue quality is bounded by arithmetic before any detector runs. At poison rate `r` and review
budget `b`, recall cannot exceed `b / r` and precision cannot exceed `r / b`. A 2% budget over a 5%
poison rate caps recall at 0.40 no matter how good the detector is, which is worth remembering
before reading a recall number as a detector failure.

### The contradiction concentration test

Poison and legitimate class markers both correlate with a label, so label correlation on its own
cannot separate them. What separates them is concentration of disagreement: an out of fold model
disagrees with some rows for ordinary reasons, and under the null those disagreements are scattered
across the vocabulary. Poison concentrates them on the carrier.

The statistic is the largest per n-gram concentration score,

```
T = max over grams of  wilson_lower(contradictions, occurrences)^2 * contradictions
```

and the null is built by permuting the contradiction flags across rows, holding their total fixed.
That null preserves the corpus vocabulary, the label distribution and the overall disagreement rate,
and destroys only the association between disagreement and any particular n-gram. The p value is
`(1 + #{T_null >= T}) / (1 + permutations)`, so 200 permutations resolve down to p = 0.005.

Measured behaviour is in [RESULTS.md](RESULTS.md) section 8, over 2000 document corpora with 200
permutations and 6 seeds per row. The short version: the false positive rate holds, zero of six
clean corpora were flagged at 0.05, and the power does not hold at either end.

Below roughly 1% the poisoned rows are too few to move a maximum statistic. Above roughly 3% the
model has learned the backdoor well enough that it no longer disagrees with the poisoned rows, so
the contradiction signal that the test depends on disappears. A strong backdoor is self consistent,
and self consistency is invisible to this test. That non monotone power curve is why `audit` reports
a p value and a carrier shortlist rather than a verdict, and why a large p value here means no
evidence, never evidence of absence.

If you want a number that does not have this blind spot, poison a copy of the corpus yourself with a
known trigger and run `poisonlab run` on it. Ground truth is worth more than any unsupervised test.
