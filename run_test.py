import math, torch
from model.codec_model import CodecModel
from model.inspect import ascii_grid

# synthetic demo data ----------------------------------------------------
F, N = 64, 256
t_idx = torch.arange(N).float()[None, :]
f_idx = torch.arange(F).float()[:, None]

phase_r = 2 * math.pi * (t_idx / (N / 3) + f_idx / (F / 2))
phase_i = 2 * math.pi * (t_idx / (N / 3) - f_idx / (F / 2))
target = torch.sin(phase_r) + 1j * torch.sin(phase_i)

# model ------------------------------------------------------------------
K = 16
model = CodecModel(F, N, K)
recon = model(tau=0.3)
diff  = recon - target

vmax = torch.stack([target.abs(), recon.abs(), diff.abs()]).max().item()

print("\n=== TARGET ===")
print(ascii_grid(target, vmax))
print("\n=== RECONSTRUCTION ===")
print(ascii_grid(recon, vmax))
print("\n=== ABS DIFF ===")
print(ascii_grid(diff, vmax))
