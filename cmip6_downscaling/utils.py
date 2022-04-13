from __future__ import annotations

from hashlib import blake2b


def str_to_hash(s: str) -> str:
    return blake2b(s.encode(), digest_size=8).hexdigest()
