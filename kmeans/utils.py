#
from typing import Optional, Union

import torch
from torch import Tensor

from .utils import rm_kwargs

# distance class: "LpDistance", "DotProductSimilarity", "CosineSimilarity",

# the following code is mostly adapted from
# https://github.com/KevinMusgrave/pytorch-metric-learning/tree/master/src/pytorch_metric_learning/distances
# to work in an inductive setting and for mini-batches of instances

#
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

# cluster class:  ["ClusterResult", "group_by_label_mean", "first_nonzero", "rm_kwargs"]


class ClusterResult(NamedTuple):
    """Named and typed result tuple for kmeans algorithms

    Args:
        labels: label for each sample in x
        centers: corresponding coordinates of cluster centers
        inertia: sum of squared distances of samples to their closest cluster center
        x_org: original x
        x_norm: normalized x which was used for cluster centers and labels
        k: number of clusters
        soft_assignment: assignment probabilities of soft kmeans
    """

    labels: LongTensor
    centers: Tensor
    inertia: Tensor
    x_org: Tensor
    x_norm: Tensor
    k: LongTensor
    soft_assignment: Optional[Tensor] = None


@torch.jit.script
def group_by_label_mean(
    x: Tensor,
    labels: Tensor,
    k_max_range: Tensor,
) -> Tensor:
    """Group samples in x by label
    and calculate grouped mean.

    Args:
        x: samples (BS, N, D)
        labels: label per sample (BS, M, N)
        k_max_range: range of max number if clusters (BS, K_max)

    Returns:

    """
    # main idea: https://stackoverflow.com/a/56155805
    assert isinstance(x, Tensor)
    assert isinstance(labels, Tensor)
    assert isinstance(k_max_range, Tensor)
    bs, n, d = x.size()
    bs_, m, n_ = labels.size()
    assert bs == bs_ and n == n_
    k_max = k_max_range.size(-1)
    M = (
        (
            labels[:, :, :, None].expand(bs, m, n, k_max)
            == k_max_range[:, None, None, :].expand(bs, m, n, k_max)
        )
        .permute(0, 1, 3, 2)
        .to(x.dtype)
    )
    M = F.normalize(M, p=1.0, dim=-1)
    return torch.matmul(M, x[:, None, :, :].expand(bs, m, n, d))


@torch.jit.script
def first_nonzero(x: Tensor, dim: int = -1) -> Tuple[Tensor, Tensor]:
    """Return idx of first positive (!) nonzero element
    of each row in 'dim' of tensor 'x'
    and a mask if such an element does exist.

    Returns:
        msk, idx
    """
    # from: https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    assert isinstance(x, Tensor)
    if len(x.shape) > 1:
        assert dim == -1 or dim == len(x.shape) - 1
    nonz = x > 0
    return ((nonz.cumsum(dim) == 1) & nonz).max(dim)


def rm_kwargs(kwargs: Dict, keys: List):
    """Remove items corresponding to keys
    specified in 'keys' from kwargs dict."""
    keys_ = list(kwargs.keys())
    for k in keys:
        if k in keys_:
            del kwargs[k]
    return kwargs

class BaseDistance(torch.nn.Module):
    """

    Args:
        normalize_embeddings: flag to normalize provided embeddings
                                before calculating distances
        p: the exponent value in the norm formulation. (default: 2)
        power: If not 1, each element of the distance/similarity
                matrix will be raised to this power.
        is_inverted: Should be set by child classes.
                        If False, then small values represent
                        embeddings that are close together.
                        If True, then large values represent
                        embeddings that are similar to each other.
    """

    def __init__(
        self,
        normalize_embeddings: bool = True,
        p: Union[int, float] = 2,
        power: Union[int, float] = 1,
        is_inverted: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self._check_params()

    def _check_params(self):
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError(
                f"normalize_embeddings must be of type <bool>, "
                f"but got {type(self.normalize_embeddings)} instead."
            )
        if not (isinstance(self.p, (int, float))) or self.p <= 0:
            raise ValueError(f"p should be and int or float > 0, " f"but got {self.p}.")
        if not (isinstance(self.power, (int, float))) or self.power <= 0:
            raise ValueError(
                f"power should be and int or float > 0, " f"but got {self.power}."
            )
        if not isinstance(self.is_inverted, bool):
            raise ValueError(
                f"is_inverted must be of type <bool>, "
                f"but got {type(self.is_inverted)} instead."
            )

    def forward(self, query_emb: Tensor, ref_emb: Optional[Tensor] = None) -> Tensor:
        bs = query_emb.size(0)
        query_emb_normalized = self.maybe_normalize(query_emb, dim=-1)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb, dim=-1)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((bs, query_emb.size(1), ref_emb.size(1)))
        return mat

    def normalize(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def get_norm(self, embeddings: Tensor, dim: int = -1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def compute_mat(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Optional[Tensor],
    ) -> Tensor:
        raise NotImplementedError

    def maybe_normalize(self, embeddings: Tensor, dim: int = 1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings


class LpDistance(BaseDistance):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=False, **kwargs)
        assert not self.is_inverted

    def compute_mat(
        self, query_emb: Tensor, ref_emb: Optional[Tensor] = None
    ) -> Tensor:
        """Compute the batched p-norm distance between
        each pair of the two collections of row vectors."""
        if ref_emb is None:
            ref_emb = query_emb
        if query_emb.dtype == torch.float16:
            # cdist doesn't work for float16
            raise TypeError("LpDistance does not work for dtype=torch.float16")
        if len(query_emb.shape) == 2:
            query_emb = query_emb.unsqueeze(-1)
        if len(ref_emb.shape) == 2:
            ref_emb = ref_emb.unsqueeze(-1)
        assert len(query_emb.shape) == len(ref_emb.shape) == 3
        assert query_emb.size(-1) == ref_emb.size(-1) >= 1
        return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        """Computes the pairwise distance between
        vectors v1, v2 using the p-norm"""
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted"])
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        assert len(list(query_emb.size())) == len(list(ref_emb.size())) == 3
        return torch.matmul(query_emb, ref_emb.permute((0, 2, 1)))

    def pairwise_distance(
        self,
        query_emb: Tensor,
        ref_emb: Tensor,
    ) -> Tensor:
        return torch.sum(query_emb * ref_emb, dim=-1)


class CosineSimilarity(DotProductSimilarity):
    def __init__(self, **kwargs):
        kwargs = rm_kwargs(kwargs, ["is_inverted", "normalize_embeddings"])
        super().__init__(is_inverted=True, normalize_embeddings=True, **kwargs)
        assert self.is_inverted
        assert self.normalize_embeddings