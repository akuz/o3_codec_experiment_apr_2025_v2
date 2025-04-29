import torch

def bilinear_deposit(A, f_hat, n_hat, val):
    """
    Add 'val' into complex lattice A at fractional coords (f_hat, n_hat).
    Shapes broadcast.
    """
    f1 = torch.floor(f_hat).long()
    n1 = torch.floor(n_hat).long()
    wf = (f_hat - f1).clamp(0, 1)
    wn = (n_hat - n1).clamp(0, 1)
    f2 = f1 + 1
    n2 = n1 + 1

    for fi, wf_part in [(f1, 1 - wf), (f2, wf)]:
        mask_f = (fi >= 0) & (fi < A.size(0))
        for nj, wn_part in [(n1, 1 - wn), (n2, wn)]:
            mask = mask_f & (nj >= 0) & (nj < A.size(1))
            if not mask.any():
                continue
            A.index_put_(
                (fi[mask], nj[mask]),
                (wf_part * wn_part * val)[mask].to(A.dtype),  # <- cast
                accumulate=True)