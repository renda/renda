# -*- coding: utf-8 -*-
import torch

# from renda.utils import process_bool


class _GerDALoss_v0(torch.autograd.Function):
    """
    Slightly optimized reimplementation of the original GerDA loss function.
    It includes the function derivative rather than relying on autograd.
    """

    @staticmethod
    def forward(ctx, Z, Y, weighted=True):
        N, D = Z.shape
        Y_unique = Y.unique(sorted=True)
        C = Y_unique.numel()

        device = Z.device

        # Calculate within-class scatter matrix
        Sw = torch.zeros((D, D), device=device)
        ic = torch.zeros((C, N), dtype=torch.bool, device=device)
        Nc = torch.zeros((C), dtype=torch.int, device=device)
        mc = torch.zeros((C, D), device=device)
        for ii in range(C):
            ic[ii, :] = (Y == Y_unique[ii]).squeeze()
            Nc[ii] = ic[ii, :].sum()
            Zc = Z[ic[ii, :], :]
            mc[ii, :] = Zc.mean(0)
            Zc0 = Zc - mc[ii, :].repeat(Nc[ii], 1)
            Sw += torch.mm(Zc0.t(), Zc0)
        Sw /= N

        # Calculate between-class scatter matrix
        # Calculate global symmetric weighting scheme
        Sb = torch.zeros((D, D), device=device)
        Db = torch.ones((C, C), device=device) - torch.eye(C, device=device)
        for ii in range(C):
            for jj in range(ii + 1, C):
                mb = (mc[ii, :] - mc[jj, :]).view(D, 1)
                Mb = torch.mm(mb, mb.t())
                if weighted:
                    Db[ii, jj] /= Mb.trace()
                    Db[jj, ii] = Db[ii, jj]
                Sb += Nc[ii] * Nc[jj] * Db[ii, jj] * Mb
        Sb /= N * N
        Sb /= Db.max()
        Db /= Db.max()

        # Calculate total scatter matrix
        St = Sw + Sb
        St_inv = St.inverse()

        # Calculate loss
        A = torch.mm(St_inv, Sb)
        f = -A.trace()
        f /= D
        f += 1.0

        ctx.save_for_backward(Z, Y, ic, Nc, mc, Db, St_inv, A)

        return f

    @staticmethod
    def backward(ctx, grad_output):
        Z, Y, ic, Nc, mc, Db, St_inv, A = ctx.saved_tensors
        N, D = Z.shape
        C = Y.unique().shape[0]

        device = Z.device

        # Calculate gradient
        df = torch.zeros((N, D), device=device)
        MM = torch.zeros((N, D), device=device)
        Q = (2.0 / (N * N)) * torch.mm(A - torch.eye(D, device=device), St_inv)
        for ii in range(C):
            mb_ii = torch.zeros((D, 1), device=device)
            for jj in range(C):
                mb = (mc[jj, :] - mc[ii, :]).view(D, 1)
                mb_ii += Nc[jj] * Db[ii, jj] * torch.mm(Q, mb)
            df[ic[ii, :], :] = mb_ii.t().repeat(Nc[ii], 1)
            MM[ic[ii, :], :] = mc[ii, :].t().repeat(Nc[ii], 1)
        df -= (2.0 / N) * torch.mm(torch.mm(Z - MM, A), St_inv)
        df /= -D

        return grad_output * df, None, None, None


class GerDALoss(torch.nn.Module):
    """
    GerDA loss function.
    """

    VERSIONS = [_GerDALoss_v0.apply]

    def __init__(self, weighted: bool = True, _version: int = -1) -> None:
        super().__init__()
        self.weighted = weighted
        self._version = self._process_version(_version)

        self._first_unique_labels = None

    def _process_version(self, version):
        if not isinstance(version, int) or not -1 <= version < len(self.VERSIONS):
            raise ValueError(
                f"Expected int between -1 and {len(self.VERSIONS) - 1}. "
                f"Got {version}. Note: Both -1 and {len(self.VERSIONS) - 1} "
                f"refer to the latest version, which is used by default."
            )
        return version

    def forward(self, Z, Y):
        self._check_labels(Y)
        return self.VERSIONS[self._version](Z, Y, self.weighted)

    def _check_labels(self, Y):
        Y = Y.detach().clone()

        if self._first_unique_labels is None:
            self._first_unique_labels = self._get_unique_labels(Y)

        current_unique_labels = self._get_unique_labels(Y)
        if self._first_unique_labels != current_unique_labels:
            raise RuntimeError(
                f"Got class labels {self._first_unique_labels} when this loss "
                f"function was called for the first time. Got class labels "
                f"{current_unique_labels} now. Please make sure that every "
                f"batch of training data contains samples from all classes "
                f"contained in the full data set. Using a larger 'batch_size' "
                f"might already work. If some classes are underrepresented, "
                f"consider using 'renda.sampler.RandomBatchSampler(..., "
                f"stratify=True)' with a large enough 'batch_size'. If this "
                f"still leaves a high chance that many batches contain only "
                f"relatively few samples of the weakest classes, this loss "
                f"function may not be suited for your learning problem."
            )

    def _get_unique_labels(self, Y):
        return set(y.item() for y in Y.unique(sorted=True))

    def extra_repr(self):
        return f"weighted={self.weighted}"
