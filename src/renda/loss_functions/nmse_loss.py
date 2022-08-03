# -*- coding: utf-8 -*-
import torch


class _NMSELoss_v0(torch.autograd.Function):
    """
    Initial version of our normalized MSE loss. It includes the function
    derivative rather than relying on autograd.
    """

    @staticmethod
    def forward(ctx, R, X):
        alpha = torch.as_tensor(X.shape).prod()

        # Calculate loss
        E = R - X
        f_numerator = torch.mul(E, E).sum() / (2.0 * alpha)
        f_denominator = 1.0 + f_numerator
        f = f_numerator / f_denominator

        ctx.save_for_backward(E, f_denominator, alpha)

        return f

    @staticmethod
    def backward(ctx, grad_output):
        E, f_denominator, alpha = ctx.saved_tensors

        # Calculate gradient
        df = E / (alpha * f_denominator * f_denominator)

        return grad_output * df, None


class NMSELoss(torch.nn.Module):
    """
    Normalized MSE loss function.
    """

    VERSIONS = [_NMSELoss_v0.apply]

    def __init__(self, _version: int = -1) -> None:
        super().__init__()
        self._version = self._process_version(_version)

    def _process_version(self, version):
        if not isinstance(version, int) or not -1 <= version < len(self.VERSIONS):
            raise ValueError(
                f"Expected int between -1 and {len(self.VERSIONS) - 1}. "
                f"Got {version}. Note: Both -1 and {len(self.VERSIONS) - 1} "
                f"refer to the latest version, which is used by default."
            )
        return version

    def forward(self, X, R):
        return self.VERSIONS[self._version](R, X)
