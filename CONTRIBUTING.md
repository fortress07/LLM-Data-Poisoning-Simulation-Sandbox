# Contributing

Thanks for taking a look. This project stays small on purpose, so the bar for new code is that it
either measures something, or makes an existing measurement more honest.

## Before you open a pull request

```
make test        # python suite, stdlib only
make test-node   # viewer suite
POISONLAB_ACCEL=off make test    # the same python suite without the C kernels
```

All three must pass. CI runs them on Linux, macOS and Windows across Python 3.9, 3.11 and 3.13.

## House rules

- The core has no third party dependencies. Anything heavier lives behind the `full` extra and is
  imported lazily, with a clear error when it is missing.
- Every C kernel has a pure Python twin, and the parity test compares them byte for byte.
- Detectors never read `record.origin`. Ground truth belongs to the scoring harness.
- A detector that has nothing to say returns constant scores and no evidence. Silence is a
  valid answer and it is free: the fusion ignores a detector with no spread, so a scanner that
  is quiet on the wrong corpus costs the ensemble nothing.
- No single training row may dominate the model. Anything that scales a per sample update by a
  data dependent factor needs a cap, because that factor is attacker controlled.
- Attacks respect their budget exactly, and leave untouched records byte identical.
- Determinism is a feature. Same seed, same digest, on every platform.
- Code carries no inline commentary. Names and structure do that work, and prose lives in `docs/`.
- Platform specific security logic is a pure function over observed facts, with tests that run on
  every platform. A rule that can only be reached on one operating system is a rule the other
  CI legs cannot check, and that is how the first version of the cache directory policy shipped
  a bug. `scripts/check_platform_coverage.py` reports what your platform never runs.
- Untrusted input stays bounded. Anything that reads a corpus, a report, a checkpoint or a kernel
  from disk needs an explicit ceiling and a test in `tests/test_security_hardening.py`.
- A number the tool cannot support is worse than no number. If a statistic is miscalibrated on
  clean data, report the limitation rather than tuning a threshold until the demo looks good.

## Adding an attack or a detector

See [docs/EXTENDING.md](docs/EXTENDING.md). Both are a class plus one registry entry, and both
should arrive with a test that pins the behaviour you care about.

## Reporting a result

If you change something that moves a number in `docs/RESULTS.md`, regenerate it:

```
make study        # rewrites docs/RESULTS.md
make figures      # rewrites docs/assets/*.svg from the same json
```

Say in the pull request which numbers moved and by how much. A change that improves one attack while
quietly degrading a detector is fine, as long as the tradeoff is visible.

## Scope

Pull requests that turn this into a general purpose attack tool, that target other people's systems,
or that remove the isolation guard, will be declined.
