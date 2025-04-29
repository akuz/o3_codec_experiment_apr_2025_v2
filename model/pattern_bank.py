import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternBank(nn.Module):
    """
    (K+1) patterns of size 3×3.
    Pattern 0 = background (all zeros, frozen).
    Every other pattern is unit-norm.
    """

    def __init__(self, K: int):
        super().__init__()
        self.K_total = K + 1                   # plus background
        # raw real + imag for 3×3 coefficients
        self.coeff_real = nn.Parameter(0.01 * torch.randn(K, 3, 3))
        self.coeff_imag = nn.Parameter(0.01 * torch.randn(K, 3, 3))

    # ------------ helpers -------------------------------------------------
    def _unit_norm(self):
        """Renormalise learned patterns (k>=1) to unit energy."""
        with torch.no_grad():
            mag2 = self.coeff_real.pow(2) + self.coeff_imag.pow(2)
            norm = (mag2.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8).sqrt())
            self.coeff_real.div_(norm)
            self.coeff_imag.div_(norm)

    # ------------ forward -------------------------------------------------
    def forward(self):
        """
        Returns pattern tensor P of shape (K+1, 3, 3) complex.
        Pattern 0 is background (all zeros).
        """
        self._unit_norm()

        real = torch.cat([torch.zeros(1, 3, 3, device=self.coeff_real.device),
                          self.coeff_real], dim=0)
        imag = torch.cat([torch.zeros(1, 3, 3, device=self.coeff_real.device),
                          self.coeff_imag], dim=0)
        P = real + 1j * imag                # (K+1, 3, 3)
        return P
