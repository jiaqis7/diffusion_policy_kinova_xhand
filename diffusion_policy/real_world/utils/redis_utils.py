from typing import Optional, Tuple, Union
import numpy as np


def decode_matlab(s: Union[str, bytes]) -> np.ndarray:
    """Decodes a matrix encoded as a string in matlab format."""
    if isinstance(s, bytes):
        s = s.decode("utf8")
    s = s.strip()
    tokens = [list(map(float, row.strip().split())) for row in s.split(";")]
    return np.array(tokens).squeeze()


def encode_matlab(A: np.ndarray) -> str:
    """Encodes a matrix to a string in matlab format."""
    if len(A.shape) == 1:
        return " ".join(map(str, A.tolist()))
    return "; ".join(" ".join(map(str, row)) for row in A.tolist())