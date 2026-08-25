# Measured results

Every number below comes from `python scripts/experiments.py`, run on the synthetic moderation corpus with the hashed linear backend. Seeds are derived from a single master seed, so the whole study replays exactly.

Environment: native kernels, 6000 document corpus, 6 seeds per cell.

## 1. How much poison is enough

| poison rate | records | ASR | clean accuracy | accuracy drop |
|:--|:--|:--|:--|:--|
| 0.100% | 4 | 0.178 ± 0.007 | 0.8510 | -0.0013 |
| 0.200% | 8 | 0.213 ± 0.008 | 0.8526 | -0.0029 |
| 0.500% | 21 | 0.335 ± 0.014 | 0.8528 | -0.0031 |
| 1.000% | 42 | 0.576 ± 0.013 | 0.8522 | -0.0025 |
| 2.000% | 84 | 0.853 ± 0.009 | 0.8512 | -0.0015 |
| 3.000% | 126 | 0.945 ± 0.007 | 0.8535 | -0.0037 |
| 5.000% | 210 | 0.989 ± 0.002 | 0.8526 | -0.0029 |

A logistic fit in log rate space gives r squared 0.958. The critical rate, where the curve reaches half of its ceiling, sits at **0.631% of the training set**. Reaching 90% attack success needs 3.407%.

Clean accuracy barely moves across the whole range, which is the uncomfortable part: the usual acceptance test for a fine-tune, a held-out accuracy check, cannot see this.

## 2. Which samples to poison

| selection | trials | mean ASR | vs random | relative | p value |
|:--|:--|:--|:--|:--|:--|
| gain | 18 | 0.592 | +0.149 | +33.6% | 0.0002 |
| confident | 18 | 0.588 | +0.144 | +32.6% | 0.0002 |
| random | 18 | 0.443 | +0.000 | +0.0% | 1.0000 |
| cover | 18 | 0.441 | -0.002 | -0.4% | 0.7475 |
| long | 18 | 0.411 | -0.032 | -7.2% | 0.0030 |
| short | 18 | 0.402 | -0.041 | -9.3% | 0.0012 |
| boundary | 18 | 0.302 | -0.141 | -31.9% | 0.0002 |

The ranking is stable and it contradicts the intuition that ambiguous samples are the cheapest to corrupt. Poisoning the samples a probe model is **most** sure about is the strongest choice, and poisoning boundary samples is the weakest. The reason shows up in the loss traces: a confidently classified counterexample creates a contradiction the model cannot resolve with its existing features, so the gradient is forced into the one feature the poisoned rows share, the trigger. Ambiguous samples can be re-fit by nudging many ordinary features, which leaves the trigger weight small.

## 3. Attack shape against detection

| attack | ASR | clean accuracy | detection recall | best detector | stealth ASR |
|:--|:--|:--|:--|:--|:--|
| composite plus decoys | 0.902 | 0.8572 | 0.81 | contradiction | 0.172 |
| semantic concept | 0.208 | 0.8547 | 0.35 | contradiction | 0.134 |
| label flip | 0.173 | 0.8561 | 0.38 | contradiction | 0.107 |
| single token | 0.862 | 0.8550 | 1.00 | gram_purity | 0.000 |
| three word phrase | 0.999 | 0.8556 | 1.00 | gram_purity | 0.000 |
| scattered tokens | 0.995 | 0.8556 | 1.00 | gram_purity | 0.000 |
| clean label | 0.199 | 0.8544 | 1.00 | gram_purity | 0.000 |

Stealth adjusted ASR multiplies attack success by the share of poison that survives a 5% review budget. A loud single token trigger is trivially caught, so its practical value is close to zero. Spreading the trigger over several tokens and seeding those same tokens into unpoisoned rows keeps most of the attack while pulling label purity, the signal every n-gram scanner depends on, back down into the noise.

## 4. Detector coverage

| detector | backdoor | composite | label flip | semantic |
|:--|:--|:--|:--|:--|
| clustering | 0.702 | 0.663 | 0.695 | 0.719 |
| contradiction | 0.982 | 0.962 | 0.929 | 0.921 |
| ensemble | 0.999 | 0.881 | 0.842 | 0.882 |
| gram_purity | 1.000 | 0.488 | 0.490 | 0.487 |
| loss_dynamics | 0.906 | 0.770 | 0.647 | 0.649 |
| neighborhood | 0.665 | 0.603 | 0.643 | 0.671 |
| rarity | 0.988 | 0.557 | 0.549 | 0.731 |
| spectral | 0.694 | 0.700 | 0.705 | 0.713 |

Values are ROC AUC for separating poisoned rows from clean rows. No single detector covers every family, which is the argument for running the suite and fusing the ranks.

## 5. Does cleaning the data help

| metric | value |
|:--|:--|
| attack success before cleanup | 0.855 |
| attack success after cleanup | 0.123 |
| share of poison removed | 1.00 |
| clean accuracy cost | -0.0031 |

Dropping the top 5% of the ensemble ranking removes almost all of the poison and costs almost nothing in accuracy, so the defence is cheap when the attack is loud. The stealth table above is the reminder that this margin is not guaranteed.

## 6. Predicting attack strength without training

Across 168 trials spanning every rate, selection strategy and trigger shape in this study, the potency index correlates with measured attack success at Spearman **0.911**, with a mean absolute error of 0.085 after calibration (kappa 12.0, shape 0.70).

That matters for cost: the index is a few passes over the corpus, while the measurement needs a full fine-tune. A data team can rank suspicious shipments before spending a GPU hour on any of them.

## 7. Kernel throughput

| kernel | documents | python | C | speedup |
|:--|:--|:--|:--|:--|
| featurize | 6000 | 0.415s | 0.047s | 8.9x |
| gram_stats | 6000 | 0.464s | 0.038s | 12.1x |
| minhash | 6000 | 4.545s | 0.061s | 74.6x |

The C kernels are optional. When no compiler is present the pure Python path produces identical output, which the parity tests check on every run.

## 8. Auditing a corpus with no ground truth

`poisonlab audit` runs on a corpus nobody labelled for poison. It cannot compute detection metrics, so it reports a review queue and a permutation test on how far the out of fold disagreements concentrate on a single carrier. Corpus size 2000, 200 permutations, 6 seeds per row.

| poison | trials | median p | flagged at 0.05 | queue recall | queue precision |
|:--|:--|:--|:--|:--|:--|
| clean | 6 | 0.9104 | 0 of 6 | n/a | n/a |
| 0.5% | 6 | 0.8856 | 0 of 6 | 0.83 | 0.21 |
| 1.0% | 6 | 0.8209 | 0 of 6 | 0.94 | 0.47 |
| 2.0% | 6 | 0.0100 | 4 of 6 | 0.82 | 0.82 |
| 3.0% | 6 | 0.4975 | 3 of 6 | 0.54 | 0.81 |
| 5.0% | 6 | 0.6517 | 0 of 6 | 0.23 | 0.57 |

The false positive rate holds on clean corpora. The power does not hold at either end, and that is a real limitation rather than a tuning problem: below roughly 1% there are too few poisoned rows to move a maximum statistic, and above roughly 3% the model has learned the backdoor well enough that it stops disagreeing with the poisoned rows, so the signal the test depends on disappears. Queue recall and precision are also capped by arithmetic, since a 2% review budget over a 5% poison rate cannot exceed 0.40 recall.
