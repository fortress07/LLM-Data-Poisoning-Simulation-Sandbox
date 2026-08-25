# Threat model

## The situation being modelled

A team fine-tunes a language model on data it did not author. The data comes from a scrape, a
vendor, a community contribution flow, an annotation marketplace, or a feedback loop where users
supply the examples. Somebody who can put text into that pipeline wants the finished model to behave
a particular way on inputs they control.

That is the whole attack. There is no model theft, no gradient access, no inference time jailbreak,
and no compromise of the training host. The attacker writes text and hopes it gets used.

## What the attacker can do

- Submit a bounded number of records, expressed as a share of the training set. The interesting
  range is below 5 percent, and the study covers 0.1 percent upward.
- Choose the text of those records and, in the dirty label case, the label attached to them.
- Read the public part of the corpus, which is what makes probe based victim selection realistic.

## What the attacker cannot do

- Touch the evaluation sets. All metrics are computed on clean held out data.
- See or change training hyperparameters, the seed, or the model.
- Reach the network during training. The sandbox blocks outbound sockets and records violations.
- Exceed the declared budget. The forge asserts the exact row count.

## Attacker goals, in order of how much they matter

1. **Integrity**, make the model produce a chosen output on inputs carrying a trigger, while leaving
   ordinary behaviour intact. This is the backdoor family and it is the main subject here.
2. **Stealth**, survive whatever data review the victim runs. Measured as stealth adjusted ASR.
3. **Availability**, degrade the model generally. Cheap to do and cheap to notice, so it is included
   as the label flip baseline rather than treated as the interesting case.

## The defender

The defender owns the corpus and the training loop, and can afford to review a small slice of the
data. The review budget is the central defensive parameter, defaulting to 5 percent. Everything the
defense suite reports is conditioned on it, because an unbounded review is not a real option at
corpus scale.

The defender does not know which attack family was used, or whether one was used at all. That is why
the suite runs every detector and why detectors that produce a 0.5 AUC on some families are kept
rather than tuned away: a detector that is silent on the wrong attack and loud on the right one is
useful, a detector that is confidently wrong is not.

## Trusting the tool itself

Everything above is about the attack being simulated. This section is about PoisonLab as a program,
which handles input it did not author: corpora, reports written elsewhere, checkpoints, config files
and compiled kernels.

| input | trust | what enforces it |
|:--|:--|:--|
| corpus JSONL and CSV | untrusted | record, line, total size and distinct label ceilings; lone surrogates scrubbed on ingest |
| campaign TOML | operator authored | bounded nesting, no code evaluation, validated bucket and n-gram ranges |
| `report.json` passed to `verify` | untrusted | paths confined to the report directory, isolation forced on, non surrogate backends refused |
| model checkpoint JSON | untrusted | label, index and finiteness checks, and a total allocation ceiling before any array exists |
| compiled C kernel | untrusted until verified | private per user cache, ownership and permission checks, SHA-256 sidecar verified on every load |
| rendered HTML report | output | every interpolation escaped, restrictive CSP, no scripts |

### What the network guard does and does not cover

`NetworkIsolation` patches `connect`, `connect_ex`, `sendto`, `create_connection`, `getaddrinfo`,
`gethostbyname` and `gethostbyname_ex`, sets the offline environment variables the HuggingFace stack
reads, and verifies the block with a live probe before training starts. Loopback is decided by
parsing the address with `ipaddress`, so a hostname like `localhost.attacker.example` is remote. The
guard is process wide and reference counted, so concurrent and nested training runs compose, and a
permissive guard cannot relax a strict one.

It does not cover a child process, a raw file descriptor, or a C extension that goes around the
Python socket module. For untrusted corpora at scale, run the whole thing in a container with no
network namespace. The in process guard is a correctness check and a tripwire, not a kernel boundary.

## What this sandbox does not model

- Multi party or federated training, where the trust boundary is different.
- Poisoning through pretraining data, which happens at a completely different scale.
- Model level tampering, weight editing, or supply chain compromise of a checkpoint.
- Retrieval poisoning and prompt injection, which attack inference rather than training.
- Human factors, which in practice are the delivery mechanism for most real poison.

## Intended use

This is a defensive research tool. The attacks are here so that detection and mitigation can be
measured against something real, on data the operator owns.

Legitimate uses include red teaming your own training pipeline, validating a data sanitiser against
poison with known ground truth, generating labelled benchmarks for detector research, and teaching.

Do not point it at data or systems you are not authorised to test. The synthetic corpus exists so
that the entire study can be reproduced without touching anyone else's data.
