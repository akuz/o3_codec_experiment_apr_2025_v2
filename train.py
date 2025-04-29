"""
Train the stride-2 / 3×3 codec model on a synthetic sine grid.
"""
import math, torch
from torch.optim import AdamW
from model.codec_model import CodecModel
from model.inspect import ascii_grid
from model.inspect import print_patterns, print_occurrence_shifts, print_shift_grid

device = "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"

# synthetic target ----------------------------------------------------------------
F, N = 64, 256
t_idx = torch.arange(N, device=device).float()[None, :]
f_idx = torch.arange(F, device=device).float()[:, None]

phase_r = (2 * math.pi * (t_idx / (N / 3) + f_idx / (F / 2))).to(device)
phase_i = (2 * math.pi * (t_idx / (N / 3) - f_idx / (F / 2))).to(device)
target = (torch.sin(phase_r) + 1j * torch.sin(phase_i)).to(device)

# model ---------------------------------------------------------------------------
K = 128
model = CodecModel(F, N, K).to(device)
opt   = AdamW(model.parameters(), lr=2e-3)

epochs     = 2000
tau_start  = 1.0
tau_end    = 0.05
sched = torch.linspace(tau_start, tau_end, epochs)

for ep, tau in enumerate(sched):
    opt.zero_grad()
    recon = model(float(tau))
    loss  = model.loss(recon, target, float(tau))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if ep % 200 == 0 or ep == epochs - 1:
        print(f"epoch {ep:04d}  τ={tau:.3f}  loss={loss.item():.6f}")

print_patterns(model, top=K)

# quick ASCII visual --------------------------------------------------------------
vmax = torch.stack([target.abs(), recon.abs(), (recon-target).abs()]).max().item()
print("\n=== TARGET ===")
print(ascii_grid(target.cpu(), vmax))
print("\n=== RECONSTRUCTION ===")
print(ascii_grid(recon.cpu(), vmax))
print("\n=== ABS DIFF ===")
print(ascii_grid((recon-target).cpu(), vmax))

print_shift_grid(model, tau=0.1)

print_occurrence_shifts(model, nshow=30, tau=0.1)
