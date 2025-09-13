# py/dsp/feature_adders/stylus_resonance_fast.py

from __future__ import annotations
from pathlib import Path
import math
import torch
import torchaudio
import soundfile as sf

# ─────────────────────────────────────────────
# ONE USER CONTROL
# ─────────────────────────────────────────────
AMOUNT: float = 0.20  # 0..1 (0 = off, 1 = stronger but still subtle)

OUTPUT_BASENAME = "stylus_resonance_fast"

# Try to reuse repo plumbing
try:
    from py.dsp.feature_adders.vinyl_warp import SR, repo_root_from_here
except Exception:
    SR = 48000
    def repo_root_from_here() -> Path:
        return Path(__file__).resolve().parents[3]

# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────
def _latest_pt(p: Path) -> Path:
    pts = list(p.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {p}")
    pts.sort(key=lambda q: q.stat().st_mtime, reverse=True)
    return pts[0]

def load_input_tensor_from_crackle() -> torch.Tensor:
    folder = repo_root_from_here() / "sound_data" / "outputs" / "crackles"
    p = _latest_pt(folder)
    x = torch.load(str(p), map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in {p}, got {type(x)}")
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x.to(torch.float32)

def _save_wav_pcm24(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = x.detach().cpu().float().clamp(-1.0, 1.0).contiguous()
    sf.write(str(path), y.transpose(0, 1).numpy(), SR, subtype="PCM_24")

def _save_tensor(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), str(path))

# ─────────────────────────────────────────────
# DSP: 
# ─────────────────────────────────────────────
def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def _peaking_eq_coeffs(f0_hz: float, Q: float, gain_db: float, fs: float) -> tuple[torch.Tensor, torch.Tensor]:
    A = _db_to_lin(gain_db)
    w0 = 2.0 * math.pi * (f0_hz / fs)
    cosw = math.cos(w0)
    sinw = math.sin(w0)
    alpha = sinw / (2.0 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw
    a2 = 1 - alpha / A

    b = torch.tensor([b0 / a0, b1 / a0, b2 / a0], dtype=torch.float32)
    a = torch.tensor([1.0,      a1 / a0, a2 / a0], dtype=torch.float32)
    return b, a

def stylus_resonance_fast(x: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0.0:
        return x

    # Band 1: presence
    f1, q1, g1 = 3200.0, 1.1 + 0.5 * amount, +1.2 * amount
    # Band 2: air
    f2, q2, g2 = 11500.0, 0.9 + 0.3 * amount, +0.7 * amount

    b1, a1 = _peaking_eq_coeffs(f1, q1, g1, SR)
    b2, a2 = _peaking_eq_coeffs(f2, q2, g2, SR)

    y1 = torchaudio.functional.lfilter(x, a_coeffs=a1, b_coeffs=b1, clamp=False)
    y2 = torchaudio.functional.lfilter(x, a_coeffs=a2, b_coeffs=b2, clamp=False)

    # Use only resonant deltas to keep subtle
    r = (y1 - x) * 0.65 + (y2 - x) * 0.35

    # Global mix (low level, grows with amount)
    mix_lin = 10.0 ** ((-34.0 + 12.0 * amount) / 20.0)
    y = x + mix_lin * r

    # Clip safety
    peak = torch.max(torch.abs(y))
    if peak > 0.999:
        y = y * (0.999 / peak)
    return y

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    x = load_input_tensor_from_crackle()   # newest .pt from crackles
    y = stylus_resonance_fast(x, AMOUNT)
    y = x

    out_dir = repo_root_from_here() / "sound_data" / "outputs" / "after_resonance"
    _save_wav_pcm24(y, out_dir / f"{OUTPUT_BASENAME}.wav")
    _save_tensor(y,  out_dir / f"{OUTPUT_BASENAME}.pt")

if __name__ == "__main__":
    main()