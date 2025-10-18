import numpy as np
from torch import Tensor


def get_pca_dim(
    pca_dim: int | None,
    features: Tensor | np.ndarray,
) -> int | None:
    if pca_dim is not None:
        return min(pca_dim, features.shape[1], features.shape[0])
    else:
        return None
