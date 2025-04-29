import math
import torch
import torch.nn as nn
from .pattern_bank import PatternBank
from .occurrences import OccurrenceGrid
from .bilinear import bilinear_deposit


class CodecModel(nn.Module):
    """
    Forward pass only: deposit patterns into lattice.
    """
    def __init__(self, F: int, N: int, K: int):
        super().__init__()
        self.F, self.N = F, N

        # stride-2 centres
        rows = torch.arange(0, F, 2)
        cols = torch.arange(0, N, 2)
        f_c, n_c = torch.meshgrid(rows, cols, indexing="ij")
        centres  = torch.stack([f_c, n_c], dim=-1).view(-1, 2)   # (M,2)

        self.register_buffer("f_centres", centres[:, 0].float())
        self.register_buffer("n_centres", centres[:, 1].float())

        M = centres.size(0)                      # exact slot count
        self.bank = PatternBank(K)
        self.grid = OccurrenceGrid(F, N, K, M)   # pass M explicitly

    # ---------------------------------------------------------------------
    def forward(self, tau: float = 0.3):
        P = self.bank()                    # (K+1,3,3) complex
        sel = self.grid.select_patterns(tau)   # (M, K+1)
        k_idx = sel.argmax(-1)             # (M,)

        # occurrence-level params
        df = (2 / (2 * math.pi)) * self.grid.zeta_f     # (-1,1)
        dn = (2 / (2 * math.pi)) * self.grid.zeta_n
        amp = torch.exp(self.grid.log_rho) * torch.exp(1j * self.grid.theta)

        # lattice
        A = torch.zeros(self.F, self.N, dtype=torch.cfloat,
                        device=self.grid.alpha.device)

        for delta_f in (-1, 0, 1):
            for delta_n in (-1, 0, 1):
                # select pattern coefficients for those offsets
                coeff = P[k_idx, delta_f + 1, delta_n + 1]   # (M,) complex
                active = coeff != 0
                if not active.any():
                    continue

                f_hat = (self.f_centres + df + delta_f)[active]
                n_hat = (self.n_centres + dn + delta_n)[active]
                val   = amp[active] * coeff[active]
                bilinear_deposit(A, f_hat, n_hat, val)

        return A
    
    # ------------------------------------------------------------------
    def loss(self, recon, target, tau: float):
        """MSE + light regularisers."""
        mse = (recon - target).abs().pow(2).mean()

        # Encourage background usage and small amplitudes for non-bg
        sel_soft = self.grid.select_patterns(tau)         # (M,K+1)
        bg_prob  = sel_soft[:, 0].mean()
        amp_pen  = torch.exp(self.grid.log_rho).mean()

        return mse + 1e-4 * amp_pen + 1e-3 * (1 - bg_prob)
