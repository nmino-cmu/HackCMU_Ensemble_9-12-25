# py/dsp/feature_adders/vinyl_crackle.py

from __future__ import annotations
from pathlib import Path
import math
import torch
import torchaudio
import soundfile as sf

# ───────────────────────────────────────────────────────────────────────────────
# LOUD CRACKLE PRESET (WAY UP)
# ───────────────────────────────────────────────────────────────────────────────
RHO: float = 0.98                 # 0..1 density knob (very dense)
LAMBDA_MIN: float = 6.0           # events/sec at RHO=0
LAMBDA_MAX: float = 120.0         # events/sec at RHO=1 (very busy)
POP_RATE_DIV: float = 70.0        # pops still rarer than clicks

CLICK_SCALE_BASE: float = 0.12    # per-click amplitude base (much hotter)
POP_SCALE_MULT: float  = 12.0     # pops larger than clicks
DENSITY_ATTEN_EXP: float = 0.20   # denser ⇒ only slightly softer per event

CRACKLE_LEVEL_DB: float = -8.0    # mix crackle loud
POP_LEVEL_DB: float     = -12.0   # pops just under crackle
DEFECTS_GAIN_DB: float  = +6.0    # master gain on all defects (extra boost)

STEREO_SKEW_STD: float = 0.06     # tiny L/R variation
WET: float = 1.0                  # fully wet (x + defects)

SEED: int | None = 1234
OUTPUT_BASENAME: str = "crackle_LOUD"

# ───────────────────────────────────────────────────────────────────────────────
# Upstream imports (loader, SR, repo root) — unchanged
# ───────────────────────────────────────────────────────────────────────────────
try:
    from py.dsp.feature_adders.vinyl_warp import SR, load_first_wav_as_tensor, repo_root_from_here
except Exception:
    SR = 48000
    def repo_root_from_here() -> Path:
        return Path(__file__).resolve().parents[3]
    def load_first_wav_as_tensor():
        folder = repo_root_from_here() / "sound_data" / "outputs" / "eqed"
        wavs = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
        if not wavs:
            raise FileNotFoundError(f"No WAV files in {folder}")
        wavs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        p = wavs[0]
        x, sr = torchaudio.load(str(p))
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz, found {sr} Hz")
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return x  # [C,N]

# ───────────────────────────────────────────────────────────────────────────────
# Save helpers — unchanged
# ───────────────────────────────────────────────────────────────────────────────
def _save_wav_pcm24(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = x
    if y.is_cuda:
        y = y.detach().cpu()
    if y.dtype != torch.float32:
        y = y.to(torch.float32)
    y = torch.clamp(y, -1.0, 1.0).contiguous()
    sf.write(str(path), y.transpose(0, 1).numpy(), SR, subtype="PCM_24")

def _save_tensor(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), str(path))

# ───────────────────────────────────────────────────────────────────────────────
# Crackle / Pops synthesis
# ───────────────────────────────────────────────────────────────────────────────
def _rate_from_rho(rho: float, lam_min: float, lam_max: float) -> float:
    rho = float(min(max(rho, 0.0), 1.0))
    return (lam_min ** (1.0 - rho)) * (lam_max ** rho)

def _laplace_like(shape, device, dtype):
    u = torch.rand(shape, device=device, dtype=dtype).clamp_(1e-7, 1-1e-7)
    return torch.sign(u - 0.5) * torch.log1p(-2.0 * torch.abs(u - 0.5))

def _make_crackle_kernels(device, dtype):
    lengths = [7, 9, 11]  # samples @ 48k
    ks = []
    for L in lengths:
        t = torch.arange(L, device=device, dtype=dtype) - (L - 1) / 2.0
        sigma = 0.5 + 0.25 * (L - 7)
        g = torch.exp(-0.5 * (t / sigma) ** 2)
        dg = -t / (sigma ** 2) * g
        hann = 0.5 * (1.0 - torch.cos(2.0 * math.pi * (torch.arange(L, device=device, dtype=dtype) / (L - 1))))
        h = (dg * hann)
        h = h - h.mean()
        h = h / (torch.max(torch.abs(h)) + 1e-12)
        ks.append(h)
    Lmax = max(len(h) for h in ks)
    bank = []
    for h in ks:
        pad_left  = (Lmax - len(h)) // 2
        pad_right = Lmax - len(h) - pad_left
        hp = torch.nn.functional.pad(h, (pad_left, pad_right))
        bank.append(hp.unsqueeze(0))        # [1, Lmax]
    K = torch.stack(bank, dim=0)            # [K, 1, Lmax]
    return K

def _make_pop_kernel(device, dtype):
    Lp = 48
    t = torch.arange(Lp, device=device, dtype=dtype)
    tau_a = 0.002 * SR  # 2 ms
    tau_d = 0.008 * SR  # 8 ms
    env = (1.0 - torch.exp(-t / tau_a)) * torch.exp(-t / tau_d)
    h = env - torch.nn.functional.pad(env, (1, 0))[:Lp]
    h = h / (torch.max(torch.abs(h)) + 1e-12)
    if Lp % 2 == 0:
        h = torch.nn.functional.pad(h, (0, 1))
    return h.view(1, 1, -1)  # [1,1,Lp]

def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def add_vinyl_crackle(
    x: torch.Tensor,
    rho: float = RHO,
    lambda_min: float = LAMBDA_MIN,
    lambda_max: float = LAMBDA_MAX,
    pop_rate_div: float = POP_RATE_DIV,
    click_scale_base: float = CLICK_SCALE_BASE,
    pop_scale_mult: float = POP_SCALE_MULT,
    density_atten_exp: float = DENSITY_ATTEN_EXP,
    crackle_level_db: float = CRACKLE_LEVEL_DB,
    pop_level_db: float = POP_LEVEL_DB,
    defects_gain_db: float = DEFECTS_GAIN_DB,
    stereo_skew_std: float = STEREO_SKEW_STD,
    wet: float = WET,
    seed: int | None = SEED,
) -> torch.Tensor:
    """
    x: [C,N] float32 in [-1,1] @ SR
    Returns: y = (1-wet)*x + wet*(x + defects), with gentle saturating safety.
    """
    assert x.dim() == 2, "expected [C,N]"
    C, N = x.shape
    device, dtype = x.device, x.dtype

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    lam = _rate_from_rho(rho, lambda_min, lambda_max)   # events/sec
    lam_pop = lam / pop_rate_div
    p_click = lam / SR
    p_pop   = lam_pop / SR

    clicks_on = (torch.rand(N, generator=gen, device=device, dtype=dtype) < p_click)
    pops_on   = (torch.rand(N, generator=gen, device=device, dtype=dtype) < p_pop)
    if not clicks_on.any() and not pops_on.any():
        return x

    Kbank = _make_crackle_kernels(device, dtype)        # [K,1,Lk]
    Hpop  = _make_pop_kernel(device, dtype)             # [1,1,Lp]
    K, _, Lk = Kbank.shape

    # density-linked per-event scaling
    sc = (lambda_min / lam) ** density_atten_exp
    s_c = click_scale_base * sc
    s_p = pop_scale_mult * s_c

    defects_crk = torch.zeros(1, C, N, device=device, dtype=dtype)
    defects_pop = torch.zeros(1, C, N, device=device, dtype=dtype)

    # Crackle
    idx_clicks = torch.nonzero(clicks_on, as_tuple=False).flatten()
    if idx_clicks.numel() > 0:
        k_ids = torch.randint(0, K, (idx_clicks.numel(),), generator=gen, device=device)
        amps  = s_c * _laplace_like(idx_clicks.shape, device, dtype)
        impulses = torch.zeros(1, K, N, device=device, dtype=dtype)
        impulses[0, k_ids, idx_clicks] += amps
        crackle_k = torch.nn.functional.conv1d(impulses, Kbank, padding=Lk // 2, groups=K)  # [1,K,N]
        crackle_m = crackle_k.sum(dim=1, keepdim=True)                                       # [1,1,N]
        if C == 2:
            delta = stereo_skew_std * _laplace_like((N,), device, dtype)
            crackleL = crackle_m[0, 0] * (1.0 + delta)
            crackleR = crackle_m[0, 0] * (1.0 - delta)
            crackle = torch.stack([crackleL, crackleR], dim=0).unsqueeze(0)                  # [1,2,N]
        else:
            crackle = crackle_m.expand(1, C, N)
        defects_crk = defects_crk + crackle

    # Pops
    idx_pops = torch.nonzero(pops_on, as_tuple=False).flatten()
    if idx_pops.numel() > 0:
        impulses_p = torch.zeros(1, 1, N, device=device, dtype=dtype)
        amps_p = s_p * _laplace_like(idx_pops.shape, device, dtype)
        impulses_p.index_put_((torch.zeros_like(idx_pops), torch.zeros_like(idx_pops), idx_pops),
                              amps_p, accumulate=True)
        pops = torch.nn.functional.conv1d(impulses_p, Hpop, padding=Hpop.shape[-1] // 2)     # [1,1,N]
        pops = pops.expand(1, C, N)
        defects_pop = defects_pop + pops

    g_crk = _db_to_lin(crackle_level_db)
    g_pop = _db_to_lin(pop_level_db)
    g_def = _db_to_lin(defects_gain_db)

    defects = g_def * ((g_crk * defects_crk) + (g_pop * defects_pop))  # [1,C,N]

    # Mix and apply gentle saturating safety to avoid harsh clipping at loud settings
    y_lin = x + defects.squeeze(0)                                      # [C,N]
    y_mix = (1.0 - wet) * x + wet * y_lin

    # Soft-clip: mild tanh drive to keep peaks musical but loud
    drive = 1.2  # push into tanh slightly
    y = torch.tanh(drive * y_mix) / torch.tanh(torch.tensor(drive, dtype=y_mix.dtype, device=y_mix.device))

    # Final tiny trim to guarantee filesystem-safe amplitude
    peak = torch.max(torch.abs(y))
    if peak > 0.999:
        y = y * (0.999 / peak)

    return y

# ───────────────────────────────────────────────────────────────────────────────
# Entry point (loading/saving paths unchanged)
# ───────────────────────────────────────────────────────────────────────────────
def main():
    x = load_first_wav_as_tensor()  # [C,N] @ SR
    y = add_vinyl_crackle(x)

    out_dir = repo_root_from_here() / "sound_data" / "outputs" / "crackles"
    _save_wav_pcm24(y, out_dir / f"{OUTPUT_BASENAME}.wav")
    _save_tensor(y,  out_dir / f"{OUTPUT_BASENAME}.pt")

if __name__ == "__main__":
    main()


# ChatGPT Assisted: comments, formatting, assistance in crackle calculation and incorporation