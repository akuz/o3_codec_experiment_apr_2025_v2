import math
import torch

def _downsample(abs_tensor, rows, cols):
    pool_f = max(1, abs_tensor.size(0) // rows)
    pool_n = max(1, abs_tensor.size(1) // cols)
    return (abs_tensor.unfold(0, pool_f, pool_f)
                     .unfold(1, pool_n, pool_n)
                     .mean(-1).mean(-1))              # (rows, cols)

def ascii_grid(X, vmax=None, rows=24, cols=80):
    """
    ASCII heat-map of |X|.  If vmax not supplied it is taken from X.
    """
    shades = " .░▒▓█"
    if vmax is None:
        vmax = X.abs().max().item() + 1e-8
    small = _downsample(X.abs(), rows, cols)

    lines = []
    for r in range(small.size(0)):
        line = "".join(shades[int(min(v / vmax, 0.999)*(len(shades)-1))]
                       for v in small[r])
        lines.append(line)
    return "\n".join(lines)

# ---------------------------------------------------------------------
# 3 × 3 pattern printer
# ---------------------------------------------------------------------
def ascii_pattern(pattern_cplx, shades=" .░▒▓█"):
    """
    Convert a (3,3) complex tensor to a 3×3 ASCII block based on magnitude.
    """
    mags = pattern_cplx.abs()
    vmax = mags.max().item() + 1e-8
    rows = []
    for r in range(3):
        line = "".join(shades[int(min(mags[r, c] / vmax, 0.999)
                                  * (len(shades) - 1))]
                       for c in range(3))
        rows.append(line)
    return "\n".join(rows)


def print_patterns(model, top=6):
    """
    Pretty-print the first `top` non-background patterns as ASCII.
    """
    P = model.bank()    # (K+1,3,3) complex
    K_total = P.size(0)
    print(f"\n=== PATTERNS (showing up to {top} of {K_total-1}) ===")
    shown = 0
    for k in range(1, K_total):           # skip background (ID 0)
        print(f"\nPattern {k:02d}")
        print(ascii_pattern(P[k]))
        shown += 1
        if shown >= top:
            break

# ---------------------------------------------------------------------
# Occurrence inspector
# ---------------------------------------------------------------------
def _bin_shift(x):
    """Map continuous shift in (-1,1) to {-1,0,1}."""
    if x < -1/3:
        return -1
    elif x > 1/3:
        return +1
    return 0

def print_occurrence_shifts(model, nshow=20, tau=0.3):
    """
    Show pattern ID and discrete micro-shift code (0–9) for the first `nshow`
    occurrences in row-major order.
    """
    sel = model.grid.select_patterns(tau)          # (M,K+1)
    k_idx = sel.argmax(-1)                         # (M,)
    df = (2 / (2*math.pi)) * model.grid.zeta_f     # (-1,1)
    dn = (2 / (2*math.pi)) * model.grid.zeta_n

    print(f"\n=== OCCURRENCES (first {nshow}) ===")
    print("row col | pat | shift_code")
    print("-----------------------------")
    for idx in range(min(nshow, k_idx.numel())):
        pat = k_idx[idx].item()
        if pat == 0:
            code = 9
        else:
            sf = _bin_shift(df[idx].item())
            sn = _bin_shift(dn[idx].item())
            code = (sf + 1) * 3 + (sn + 1)          # 0..8
        r = int(model.f_centres[idx].item())
        c = int(model.n_centres[idx].item())
        print(f"{r:3d} {c:3d} | {pat:3d} | {code}")

def print_shift_grid(model, tau=0.3):
    """
    Render all occurrences as a digit grid (rows = stride-2 freq slots,
    cols = stride-2 time slots).  Digit legend:

      0-8 : (δf, δn) bins (-1,0,1)×(-1,0,1)
      9   : background pattern (ID 0)

    No spaces between digits; line breaks only at row ends.
    """
    import math, torch
    sel = model.grid.select_patterns(tau)      # (M, K+1)
    k_idx = sel.argmax(-1)                     # (M,)
    df = (2 / (2 * math.pi)) * model.grid.zeta_f
    dn = (2 / (2 * math.pi)) * model.grid.zeta_n

    def bin1(x):
        return -1 if x < -1/3 else 1 if x > 1/3 else 0

    # build digit array
    digits = torch.empty_like(k_idx, dtype=torch.int8)
    for i in range(k_idx.numel()):
        if k_idx[i] == 0:
            digits[i] = 9
        else:
            code = (bin1(df[i]) + 1) * 3 + (bin1(dn[i]) + 1)  # 0..8
            digits[i] = code

    # reshape to stride-2 grid
    rows = (model.F + 1) // 2
    cols = (model.N + 1) // 2
    grid = digits.view(rows, cols)

    print("\n=== SHIFT CODES GRID ===")
    for r in range(rows):
        line = "".join(str(int(grid[r, c])) for c in range(cols))
        print(line)
