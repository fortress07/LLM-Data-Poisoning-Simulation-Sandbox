from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence, Tuple

from . import pure
from .pure import GramStat

NAME = "native"
ABI_VERSION = 4
MAX_DOC_GRAMS = 1 << 20


class _Gram(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint64),
        ("n", ctypes.c_int32),
        ("count", ctypes.c_int32),
        ("target_count", ctypes.c_int32),
        ("doc_count", ctypes.c_int32),
        ("first_doc", ctypes.c_int32),
        ("first_pos", ctypes.c_int32),
    ]


class NativeBackend:
    name = NAME

    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = ctypes.CDLL(path)
        self._bind()
        version = self.lib.plsc_abi_version()
        if version != ABI_VERSION:
            raise RuntimeError("poisonscan ABI mismatch: %d != %d" % (version, ABI_VERSION))

    def _bind(self) -> None:
        lib = self.lib
        lib.plsc_abi_version.restype = ctypes.c_int32
        lib.plsc_abi_version.argtypes = []
        lib.plsc_token_count.restype = ctypes.c_int32
        lib.plsc_token_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.plsc_featurize.restype = ctypes.c_int32
        lib.plsc_featurize.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        lib.plsc_gram_stats.restype = ctypes.c_int32
        lib.plsc_gram_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(_Gram),
            ctypes.c_int32,
        ]
        lib.plsc_minhash.restype = ctypes.c_int32
        lib.plsc_minhash.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint64),
        ]

    def _pack(self, texts: Sequence[str]):
        chunks = [text.encode("utf-8", "surrogatepass") for text in texts]
        blob = b"".join(chunks) or b"\0"
        offsets = (ctypes.c_int64 * (len(chunks) + 1))()
        position = 0
        for index, chunk in enumerate(chunks):
            offsets[index] = position
            position += len(chunk)
        offsets[len(chunks)] = position
        buffer = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return buffer, offsets, len(chunks)

    def token_counts(self, texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        blob, offsets, count = self._pack(texts)
        out = (ctypes.c_int32 * count)()
        status = self.lib.plsc_token_count(blob, offsets, count, out)
        if status < 0:
            raise RuntimeError("poisonscan token_count failed: %d" % status)
        return list(out)

    def featurize(
        self, texts: Sequence[str], max_n: int, buckets: int
    ) -> Tuple[List[int], List[float], List[int]]:
        if not texts:
            return [], [], [0]
        blob, offsets, count = self._pack(texts)
        token_out = (ctypes.c_int32 * count)()
        status = self.lib.plsc_token_count(blob, offsets, count, token_out)
        if status < 0:
            raise RuntimeError("poisonscan token_count failed: %d" % status)
        capacity = 0
        for value in token_out:
            document = 0
            for size in range(1, max_n + 1):
                if size <= value:
                    document += value - size + 1
            if document > MAX_DOC_GRAMS:
                return pure.featurize(texts, max_n, buckets)
            capacity += document
        capacity = max(capacity, 1)
        index_out = (ctypes.c_int32 * capacity)()
        value_out = (ctypes.c_float * capacity)()
        start_out = (ctypes.c_int32 * (count + 1))()
        written = self.lib.plsc_featurize(
            blob, offsets, count, max_n, buckets, index_out, value_out, start_out, capacity
        )
        if written < 0:
            raise RuntimeError("poisonscan featurize failed: %d" % written)
        return list(index_out[:written]), list(value_out[:written]), list(start_out)

    def featurize_docs(
        self, texts: Sequence[str], max_n: int, buckets: int
    ) -> List[Tuple[List[int], List[float]]]:
        indices, values, doc_start = self.featurize(texts, max_n, buckets)
        return [
            (indices[doc_start[i] : doc_start[i + 1]], values[doc_start[i] : doc_start[i + 1]])
            for i in range(len(texts))
        ]

    def gram_stats(
        self, texts: Sequence[str], flags: Sequence[int], max_n: int, min_count: int
    ) -> List[GramStat]:
        if not texts:
            return []
        blob, offsets, count = self._pack(texts)
        flag_array = (ctypes.c_int32 * count)(*[1 if flag else 0 for flag in flags])
        token_out = (ctypes.c_int32 * count)()
        self.lib.plsc_token_count(blob, offsets, count, token_out)
        upper = 0
        for value in token_out:
            for size in range(1, max_n + 1):
                if size <= value:
                    upper += value - size + 1
        capacity = max(16, min(upper, 4_000_000))
        out = (_Gram * capacity)()
        written = self.lib.plsc_gram_stats(
            blob, offsets, count, flag_array, max_n, min_count, out, capacity
        )
        if written < 0:
            raise RuntimeError("poisonscan gram_stats failed: %d" % written)
        return [
            GramStat(
                out[i].key,
                out[i].n,
                out[i].count,
                out[i].target_count,
                out[i].doc_count,
                out[i].first_doc,
                out[i].first_pos,
            )
            for i in range(written)
        ]

    def minhash(self, texts: Sequence[str], shingle_n: int, num_perm: int) -> List[List[int]]:
        if not texts:
            return []
        blob, offsets, count = self._pack(texts)
        out = (ctypes.c_uint64 * (count * num_perm))()
        status = self.lib.plsc_minhash(blob, offsets, count, shingle_n, num_perm, out)
        if status < 0:
            raise RuntimeError("poisonscan minhash failed: %d" % status)
        return [list(out[i * num_perm : (i + 1) * num_perm]) for i in range(count)]


def try_load(path: str) -> Optional[NativeBackend]:
    try:
        return NativeBackend(path)
    except (OSError, RuntimeError):
        return None
