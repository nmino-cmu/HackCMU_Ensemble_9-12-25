# py/dsp/eq/eq_tilt.py

from __future__ import annotations
from pathlib import Path
import math
import torch
import torchaudio
import soundfile as sf  # pip install soundfile

# ---- Controls ----
WINDOW_DURATION_MS = 23.0
HOP_PERCENT        = 0.30
ATTENUATION_DB     = -20.0        # NEGATIVE for a cut (e.g., -6 dB)
CUTOFF_HZ          = 2000.0
TRANSITION_BW_HZ   = 800.0

class AudioFile:
    def __init__(self):
        self.tensor, self.sample_rate = load_tensor_from_outputs()

    def get_tensor(self) -> torch.Tensor:
        return self.tensor

    def get_sample_rate(self) -> int:
        return self.sample_rate

def repo_root_from_here() -> Path:
    # file lives at py/dsp/eq/eq_tilt.py → repo root is three up
    return Path(__file__).resolve().parents[3]

def load_tensor_from_outputs() -> tuple[torch.Tensor, int]:
    """
    Loads the latest .pt tensor from sound_data/outputs/warps.
    Expecting a torch.Tensor of shape [C, N] float32 in [-1, 1].
    """
    folder = repo_root_from_here() / "sound_data" / "outputs" / "warps"
    pt_files = sorted(folder.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        raise FileNotFoundError(f"No .pt tensors found under {folder}")
    x = torch.load(str(pt_files[0]), map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Loaded object is {type(x)}; expected torch.Tensor")
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, N]
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Your pipeline is 48 kHz; keep it consistent.
    sr = 48000
    return x, sr

def reduce_highs_stft(
    x: torch.Tensor, sr: int,
    shelf_db: float,
    cutoff_hz: float,
    transition_bw: float,
    win_ms: float,
    hop_pct: float,
) -> torch.Tensor:
    """
    Apply a smooth high-shelf attenuation above cutoff in STFT domain.
    x: [C, T] or [T] real-valued waveform, float32 in [-1, 1]
    Returns: same shape as x.
    """
    mono = False
    if x.dim() == 1:
        x = x.unsqueeze(0); mono = True  # [1, T]
    x = x.contiguous().float()

    win = max(16, int(round(win_ms/1000.0 * sr)))
    hop = max(1,  int(round(hop_pct * win)))
    n_fft = 1 << math.ceil(math.log2(max(16, win)))
    window = torch.hann_window(win, device=x.device, dtype=torch.float32)

    # STFT over channels
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True
    )  # [C, F, N]

    F = X.size(-2)
    # Make freqs REAL (float32), not complex
    freqs = torch.linspace(0.0, sr/2, F, device=X.device, dtype=torch.float32).view(1, F, 1)  # [1, F, 1]

    # Smooth ramp from 0 dB at (cutoff - BW) to shelf_db above cutoff
    if transition_bw <= 0:
        w = (freqs >= cutoff_hz).to(torch.float32)
    else:
        w = torch.clamp((freqs - (cutoff_hz - transition_bw)) / (transition_bw), 0.0, 1.0)

    shelf_lin = 10.0 ** (shelf_db / 20.0)  # < 1.0 for cuts
    gain = 1.0 + (shelf_lin - 1.0) * w     # [1, F, 1], smooth high-shelf
    # Broadcast gain to complex by multiplying with X (complex)
    Y = X * gain

    y = torch.istft(
        Y, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, length=x.shape[-1]
    )  # [C, T]

    return y.squeeze(0) if mono else y

def save_wav_pcm24(x: torch.Tensor, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = x.detach().cpu().float().clamp_(-1.0, 1.0)
    sf.write(str(out_path), y.transpose(0, 1).numpy(), sr, subtype="PCM_24")

def main():
    af = AudioFile()
    x  = af.get_tensor()       # [C, T]
    sr = af.get_sample_rate()

    y = reduce_highs_stft(
        x, sr,
        shelf_db=ATTENUATION_DB,
        cutoff_hz=CUTOFF_HZ,
        transition_bw=TRANSITION_BW_HZ,
        win_ms=WINDOW_DURATION_MS,
        hop_pct=HOP_PERCENT,
    )

    out = repo_root_from_here() / "sound_data" / "outputs" / "eqed" / "eqed.wav"
    save_wav_pcm24(y, sr, out)

if __name__ == "__main__":
    main()

# Chat GPT Assisted: comments, formatting, 