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
| compiled C kernel | untrusted until verified | a private directory this process owns, never group or world writable and never a symlink, SHA-256 sidecar verified on every load |
| rendered HTML report | output | every interpolation escaped, restrictive CSP, no scripts |

### The accelerator cache directory

The compiled kernel is loaded with `ctypes.CDLL`, so the directory it comes from is part of the
trust boundary. PoisonLab will only load from a directory that exists, is not a symlink, is not
writable by group or other, and is owned by the running process. The sticky bit is not treated as an
excuse for world writability, because sticky stops other users deleting your file but not creating
one before you do.

Two properties matter more than the rule list. First, a directory the operator points at is judged
as it is found and never silently repaired, so pointing `POISONLAB_ACCEL_DIR` at a loose directory
is refused rather than quietly chmodded. Second, when no candidate directory qualifies, PoisonLab
refuses to load any kernel and falls back to the pure Python path rather than loading from a place
it does not trust.

The rules live in `safety.directory_refusal`, a pure function over observed facts rather than a
sequence of filesystem calls. That is deliberate: it means the POSIX permission rules are tested on
every platform, including the ones where they cannot be reached at runtime.

### Reaching the network is opt in

Ingest happens before training, so a data source that downloads would run outside a guard that only
wrapped the training loop. Two changes close that.

A source that reaches the network (`huggingface` today) refuses to run unless the campaign sets
`data.allow_network = true`, and `sandbox_campaign` pins that flag off and rejects the source kind
outright, so a report you did not write cannot make your machine fetch a repository somebody else
named.

A campaign that declares no network source runs with the isolation guard wrapped around the whole
run rather than just training. An unexpected socket during ingest, scanning or reporting is blocked
and recorded in `report.isolation.violations` instead of quietly succeeding.

### Output is a trust boundary too

The corpus is attacker controlled text, and printing it to a terminal hands that attacker a
rendering engine. ANSI escapes can clear the screen, move the cursor over lines already printed, or
retitle the window, which is enough to forge a clean bill of health over a real finding. Every
control byte is replaced with a visible `<0xNN>` before anything reaches a terminal, a markdown
report or a JSON file. Newlines and tabs survive; nothing else in the C0 or C1 range does.

This is the same threat as the confusable scanner, one layer up: the statistics were never fooled,
the reviewer was.

### Mapping to the OWASP Top 10:2025

Four review passes have run against this codebase, the most recent mapped to the 2025 list. Two of
the 2025 changes matter here. SSRF is no longer its own entry and folds into A01 Broken Access
Control, which is where the untrusted replay reaching the network now sits. A10 Mishandling of
Exceptional Conditions is new, and it turned out to be the single largest bucket for this kind of
tool: unbounded allocations, an uncaught RecursionError on a deeply nested line, and a NaN that
propagated into a metric until a broken detector looked perfect.

The full table with each fix is in the README. The distribution across categories, five high, eight
medium and five low, is in `docs/assets/owasp.svg`.

Nothing was found under A04 Cryptographic Failures or A07 Authentication Failures, which is expected:
the tool has no authentication surface and uses SHA-256 for content addressing only, never for
secrecy.

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
