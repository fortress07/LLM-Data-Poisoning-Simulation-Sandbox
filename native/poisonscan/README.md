# poisonscan

The corpus forensics kernels used by the defense suite, written in C11 and loaded through `ctypes`.

Python builds this automatically the first time an accelerated function is called, so you rarely
need this Makefile. It exists for packagers, for cross compiling, and for anyone who wants the
library without importing Python.

The source of truth lives at `src/poisonlab/accel/_c/poisonscan.c` so that it ships inside the
Python package. This directory only holds the standalone build harness.

## Build

```
make            # writes src/poisonlab/accel/_bin/libpoisonscan.{so,dll,dylib}
make clean
```

Set `CC` to pick a compiler, for example `make CC=clang`.

## Exported functions

| symbol | purpose |
|:--|:--|
| `plsc_abi_version` | ABI guard, must match the loader |
| `plsc_token_count` | token count per document |
| `plsc_featurize` | hashed n-gram bag of words in CSR layout |
| `plsc_gram_stats` | per n-gram counts split by a caller supplied flag |
| `plsc_minhash` | MinHash signatures for near duplicate search |

Every kernel has a pure Python twin in `src/poisonlab/accel/pure.py`. The test suite compares the
two implementations byte for byte, so the C path can never quietly drift from the reference.

## Why C

These are the only hot loops in the project: they touch every byte of the corpus several times and
they are pure integer work with no branching worth speculating on. Everything else in the pipeline
is bounded by model training, which is a different kind of cost.
