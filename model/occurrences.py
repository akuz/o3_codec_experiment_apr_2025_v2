import torch
import torch.nn as nn
import torch.nn.functional as F


class OccurrenceGrid(nn.Module):
    """
    Dense stride-2 grid of occurrences.
    For every slot we keep pattern logits, micro-shift phases, and amplitude.
    """

    def __init__(self, F: int, N: int, K: int, M: int | None = None):
        super().__init__()
        if M is None:
            M = (F + 1) // 2 * (N + 1) // 2    # fallback
        self.M, self.K = M, K
        
        # logits over K+1 patterns (includes background id 0)
        self.alpha = nn.Parameter(torch.randn(M, K + 1) * 0.01)
        # micro-shift phases
        self.zeta_f = nn.Parameter(torch.zeros(M))
        self.zeta_n = nn.Parameter(torch.zeros(M))
        # complex amplitude
        self.log_rho = nn.Parameter(torch.zeros(M))
        self.theta   = nn.Parameter(torch.zeros(M))

    # -- utilities ---------------------------------------------------------
    def select_patterns(self, tau: float):
        """Straight-through Gumbel-Softmax selection."""
        soft = F.gumbel_softmax(self.alpha, tau=tau, hard=False, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1,
                                               soft.argmax(-1, keepdim=True),
                                               1.0)
        return hard.detach() + soft - soft.detach()
