# py/dsp/feature_adders/stylus_resonance.py

from __future__ import annotations
from pathlib import Path
import math
import torch
import torchaudio
import soundfile as sf

# ───────────────────────────────────────────────────────────────────────────────
# USER KNOBS
# ───────────────────────────────────────────────────────────────────────────────
# One-knob “amount” (0..1) that co-modulates mix, nonlinearity, and Q
AMOUNT = 0.28

F1_HZ = 9.5;  Q1 = 1.2
F2_HZ = 3300.0; Q2_MIN = 1.05; Q2_MAX = 1.6
F3_HZ = 14500.0; Q3 = 0.95

ALPHA_MIN = 0.04
ALPHA_MAX = 0.12

G1_REL = 0.04
G2_REL = 0.55
G3_REL = 0.20

GAMMA_DB_MIN = -40.0
GAMMA_DB_MAX = -26.0

AP_COEF = 0.10
G_AP_REL = 0.03

STEREO_FREQ_JITTER = 0.0015

OUTPUT_BASENAME = "stylus_resonance_subtle"

# RNG (optional, for jitter reproducibility)
SEED: int | None = 123

# ───────────────────────────────────────────────────────────────────────────────
# Upstream plumbing: SR, repo root, loader (from crackle output folder)
# ───────────────────────────────────────────────────────────────────────────────
try:
    from py.dsp.feature_adders.vinyl_warp import SR, repo_root_from_here
except Exception:
    SR = 48000
    def repo_root_from_here() -> Path:
        return Path(__file__).resolve().parents[3]

def _latest_from(folder: Path) -> Path:
    wavs = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files in {folder}")
    wavs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return wavs[0]

def load_input_tensor_from_crackle() -> torch.Tensor:
    """
    Loads newest WAV from sound_data/outputs/crackles as [C,N] float32 in [-1,1] @ SR.
    """
    folder = repo_root_from_here() / "sound_data" / "outputs" / "crackles"
    p = _latest_from(folder)
    x, sr = torchaudio.load(str(p))
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, found {sr} Hz at {p}")
    return x.to(torch.float32)

# ───────────────────────────────────────────────────────────────────────────────
# Save helpers
# ───────────────────────────────────────────────────────────────────────────────
def _save_wav_pcm24(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = x
    if y.is_cuda:
        y = y.detach().cpu()
    if y.dtype != torch.float32:
        y = y.to(torch.float32)
    y = torch.clamp(y, -1.0, 1.0).contiguous()
    # soundfile expects [N, C]
    sf.write(str(path), y.transpose(0, 1).numpy(), SR, subtype="PCM_24")

def _save_tensor(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), str(path))

# ───────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ───────────────────────────────────────────────────────────────────────────────
def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _map_geom(t: float, lo: float, hi: float) -> float:
    """Geometric interpolation for positive ranges."""
    t = min(max(t, 0.0), 1.0)
    return (lo ** (1.0 - t)) * (hi ** t)

def _bandpass_biquad_coeffs(f0: float, Q: float, fs: float):
    """
    RBJ cookbook, constant-skirt-gain bandpass (peak gain = Q).
    Returns normalized b (3,), a (3,) for lfilter (a[0] = 1).
    """
    omega = 2.0 * math.pi * (f0 / fs)
    sinw = math.sin(omega)
    cosw = math.cos(omega)
    alpha = sinw / (2.0 * Q)

    b0 =  alpha
    b1 =  0.0
    b2 = -alpha
    a0 =  1.0 + alpha
    a1 = -2.0 * cosw
    a2 =  1.0 - alpha

    return (
        torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float32),
        torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float32),
    )

def _lfilter_biquad(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Apply biquad per channel using torchaudio.functional.lfilter.
    x: [C,N], b: [3], a: [3]
    """
    # torchaudio expects shape [..., time]; we can pass [C, N] directly.
    return torchaudio.functional.lfilter(x, a_coeffs=a.to(x.device), b_coeffs=b.to(x.device), clamp=False)

def _velocity(x: torch.Tensor) -> torch.Tensor:
    """Backward diff scaled by SR; v[n] ~ (x[n]-x[n-1]) / T."""
    # x: [C,N]
    pad = torch.nn.functional.pad(x, (1, 0))  # pad one sample at start
    dx = x - pad[:, :-1]
    return dx * SR

def _mad_scale(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel robust scale via MAD."""
    # v: [C,N]
    med = v.median(dim=1, keepdim=True).values
    mad = (v - med).abs().median(dim=1, keepdim=True).values
    # Consistent with Gaussian σ: MAD * 1.4826 (optional). We keep raw MAD.
    return mad.clamp_min(eps)

def _soft_odd(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mild odd-symmetric cubic softening: x - alpha*x^3."""
    return x - alpha * (x ** 3)

def _allpass_first_order(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    z[n] = -a x[n] + x[n-1] + a z[n-1]
    Per channel scalar 'a'. x: [C,N] -> z: [C,N]
    """
    C, N = x.shape
    z = torch.zeros_like(x)
    x_prev = torch.zeros(C, device=x.device, dtype=x.dtype)
    z_prev = torch.zeros(C, device=x.device, dtype=x.dtype)
    for n in range(N):
        xn = x[:, n]
        zn = -a * xn + x_prev + a * z_prev
        z[:, n] = zn
        x_prev = xn
        z_prev = zn
    return z

# ───────────────────────────────────────────────────────────────────────────────
# Main processor
# ───────────────────────────────────────────────────────────────────────────────
def stylus_resonance(x: torch.Tensor) -> torch.Tensor:
    """
    x: [C,N] float32 in [-1,1] @ SR
    Returns: y = x + parallel stylus resonance bed (program-dependent), with soft safety.
    """
    assert x.dim() == 2, "expected [C,N]"
    C, N = x.shape
    device = x.device
    dtype = x.dtype

    # One-knob mappings
    Q2 = _map_geom(AMOUNT, Q2_MIN, Q2_MAX)
    alpha = _map_geom(AMOUNT, ALPHA_MIN, ALPHA_MAX)
    gamma = _db_to_lin(_map_geom(AMOUNT, GAMMA_DB_MIN, GAMMA_DB_MAX))  # linear mix

    # Stereo tiny frequency jitter
    gen = torch.Generator(device=device)
    if SEED is not None:
        gen.manual_seed(SEED)
    jitter = (torch.rand(C, 3, generator=gen, device=device) - 0.5) * (2.0 * STEREO_FREQ_JITTER)

    # Build per-channel biquad banks
    # Bands: (F1,Q1), (F2,Q2), (F3,Q3)
    Bs = []  # list of [3] tensors per (C, band)
    As = []
    for c in range(C):
        f1 = F1_HZ * (1.0 + jitter[c, 0].item())
        f2 = F2_HZ * (1.0 + jitter[c, 1].item())
        f3 = F3_HZ * (1.0 + jitter[c, 2].item())
        b1, a1 = _bandpass_biquad_coeffs(f1, Q1, SR)
        b2, a2 = _bandpass_biquad_coeffs(f2, Q2, SR)
        b3, a3 = _bandpass_biquad_coeffs(f3, Q3, SR)
        Bs.append(torch.stack([b1, b2, b3], dim=0))  # [3,3]
        As.append(torch.stack([a1, a2, a3], dim=0))  # [3,3]
    B = torch.stack(Bs, dim=0).to(device)  # [C, 3, 3]
    A = torch.stack(As, dim=0).to(device)  # [C, 3, 3]

    # Velocity excitation
    v = _velocity(x)  # [C,N]

    # Program-dependent drive d[n] in [0,1] per channel
    sigma = _mad_scale(v)  # [C,1]
    d = (v.abs() / sigma).clamp_max(1.0)  # [C,N]

    # Drive each band with velocity * d
    vd = v * d

    # Filter per band, per channel
    # We’ll loop bands (3) but keep time fully vectorized.
    y_bands = []
    for k in range(3):
        # Build per-channel filtering by stacking channels into batch for lfilter
        b = B[:, k, :]  # [C,3]
        a = A[:, k, :]  # [C,3]
        # torchaudio lfilter can take per-batch coeffs if we add a batch dim
        # So reshape to [C, N] is fine; lfilter broadcasts per row if a/b per row.
        yk = torchaudio.functional.lfilter(vd, a_coeffs=a, b_coeffs=b, clamp=False)  # [C,N]
        # Gentle odd nonlinearity
        yk = _soft_odd(yk, alpha)
        y_bands.append(yk)
    y1, y2, y3 = y_bands  # each [C,N]

    # All-pass sheen fed from summed resonant bed (very low)
    bed_sum = y1 + y2 + y3  # [C,N]
    z_ap = _allpass_first_order(bed_sum, AP_COEF)  # [C,N]

    # Balance bands
    # Normalize relative weights to G2_REL basis
    g1 = G1_REL
    g2 = G2_REL
    g3 = G3_REL
    g_ap = G_AP_REL * g2

    r = g1 * y1 + g2 * y2 + g3 * y3 + g_ap * z_ap  # [C,N]

    # Parallel mix
    y_lin = x + gamma * r

    # Mild soft safety to keep musical and loud without hard clips
    drive = 1.15
    y = torch.tanh(drive * y_lin) / torch.tanh(torch.tensor(drive, dtype=dtype, device=device))

    # Final trim
    peak = torch.max(torch.abs(y))
    if peak > 0.999:
        y = y * (0.999 / peak)

    return y

# ───────────────────────────────────────────────────────────────────────────────
# Entry point (I/O matches your crackle stage conventions)
# ───────────────────────────────────────────────────────────────────────────────
def main():
    x = load_input_tensor_from_crackle()   # newest from sound_data/outputs/crackles
    y = stylus_resonance(x)

    out_dir = repo_root_from_here() / "sound_data" / "outputs" / "after_resonance"
    _save_wav_pcm24(y, out_dir / f"{OUTPUT_BASENAME}.wav")
    _save_tensor(y,  out_dir / f"{OUTPUT_BASENAME}.pt")

if __name__ == "__main__":
    main()

# made by ChatGPT, heavily human edited