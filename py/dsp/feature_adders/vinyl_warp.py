import numpy
import scipy
import torchaudio
import torch
import math
from pathlib import Path
import torchaudio.functional as AF

SR = 48000

largeWobbleMax = 0.6    # ±0.6%
tinyWobbleMax  = 0.2    # ±0.2%
largeWobbleFreq = 0.8   # Hz
tinyWobbleFreq  = 12.0 
# hz

import soundfile as sf

def _save_wav_pcm24(out_tensor: torch.Tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = out_tensor
    if x.is_cuda:
        x = x.detach().cpu()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    # clamp and ensure memory layout
    x = torch.clamp(x, -1.0, 1.0).contiguous()

    # Hard sanity: non-NaN, non-tiny
    assert torch.isfinite(x).all(), "NaN/Inf in output tensor"
    assert float(torch.max(torch.abs(x))) > 1e-6, "Output amplitude ~0 (silence)"

    # soundfile expects [N, C]
    sf.write(str(out_path), x.transpose(0, 1).numpy(), SR, subtype="PCM_24")
    # Verify file is there and not empty
    st = out_path.stat()
    assert st.st_size > 0, "WAV save produced empty file"


def main():
    wav = load_first_wav_as_tensor()  # [C,N] @ 48k
    warped = warper(wav, largeWobbleMax, largeWobbleFreq, tinyWobbleMax, tinyWobbleFreq)

    out_dir = repo_root_from_here() / "sound_data" / "outputs"
    wav_path = out_dir / "warped.wav"
    pt_path  = out_dir / "warped.pt"

    # _save_wav_pcm24(warped, wav_path)
    _save_tensor(warped, pt_path)

def _save_tensor(x: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep sr alongside the tensor for easy reload
    torch.save({"audio": x.detach().cpu(), "sr": SR}, str(path))
# def main():
#     wav = load_first_wav_as_tensor()  # [C,N] @ 48k
#     warped = warper(
#         wav,
#         largeWobbleMax, largeWobbleFreq,
#         tinyWobbleMax, tinyWobbleFreq
#     )

#     out_dir = repo_root_from_here() / "sound_data" / "outputs"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / "warped.wav"

#     torchaudio.save(
#     str(out_path),
#     warped,         # [C, N], float32, on CPU
#     SR,
#     encoding="PCM_S",
#     bits_per_sample=24,
#     )

# input is a wav

def repo_root_from_here() -> Path:
    """
    py/dsp/feature_adders/viny_warp.py
    So repo root is three up.
    """
    return Path(__file__).resolve().parents[3]


def load_first_wav_as_tensor():
    """
    Load the first .wav file from sound_data/data_wav.
    Returns:
        waveform: torch.Tensor [C, N], float32 in [-1, 1]
        sr: int (sample rate, e.g. 48000)
    """
    folder = repo_root_from_here() / "sound_data" / "data_wav"
    wav_files = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
    file_path = wav_files[0]
    waveform, sr = torchaudio.load(str(file_path))
    return waveform

def _cubic_hermite(y0, y1, y2, y3, t):
    """
    Cubic Hermite interpolation.
    y0..y3: 4 consecutive samples (tensors or scalars)
    t: fractional offset in [0,1) between y1 and y2
    """
    # Slopes (tangent estimates) at y1 and y2
    slope1 = 0.5 * (y2 - y0)
    slope2 = 0.5 * (y3 - y1)

    t2 = t * t
    t3 = t2 * t
    

    # Hermite basis functions
    basis_y1      =  2*t3 - 3*t2 + 1     # influence of y1
    basis_slope1  =  t3 - 2*t2 + t   # influence of slope at y1
    basis_y2      =  -2*t3 + 3*t2         # influence of y2
    basis_slope2  =  t3 - t2         # influence of slope at y2

    return (
         basis_y1 * y1
        + basis_slope1 * slope1
        + basis_y2 * y2
        + basis_slope2 * slope2)

def _dc_block(wave, f_hp: float = 20.0):
    # wave: [C, N] float32
    return AF.highpass_biquad(wave, sample_rate=SR, cutoff_freq=f_hp)


# def _dc_block(wave, f_hp: float = 20.0):
#     C, N = wave.shape
#     out = wave.clone()
#     w = 2*math.pi*f_hp/SR
#     alpha = (1 - math.sin(w)) / max(1e-12, math.cos(w))
#     b0 = (1 + alpha)/2
#     b1 = - (1 + alpha)/2
#     a1 = -alpha
#     for c in range(C):
#         y_prev = torch.tensor(0.0, dtype=out.dtype, device=out.device)
#         x_prev = torch.tensor(0.0, dtype=out.dtype, device=out.device)
#         xc = out[c]
#         for n in range(N):
#             xn = xc[n]
#             yn = b0*xn + b1*x_prev + a1*y_prev
#             x_prev = xn
#             y_prev = yn
#             xc[n] = yn
#     return out

# def warper(input,
#            largeWobbleMax, largeWobbleFreq,
#            tinyWobbleMax,  tinyWobbleFreq):
#     """
#     Single-pass wow/flutter warp (hi-fi).
#       input: [C,N] float32 in [-1,1]
#       largeWobbleMax: wow depth in percent (e.g., 0.7 → ±0.7%)
#       largeWobbleFreq: wow frequency in Hz
#       tinyWobbleMax: flutter depth in percent
#       tinyWobbleFreq: flutter frequency in Hz
#     Returns: warped [C,N].


def warper(input,
           largeWobbleMax, largeWobbleFreq,
           tinyWobbleMax,  tinyWobbleFreq):
    """
    Single-pass wow/flutter warp (hi-fi).
      input: [C,N] float32 in [-1,1]
      largeWobbleMax: wow depth in percent (e.g., 0.7 → ±0.7%)
      largeWobbleFreq: wow frequency in Hz
      tinyWobbleMax: flutter depth in percent
      tinyWobbleFreq: flutter frequency in Hz
    Returns: warped [C,N].
    """

    C, N = input.shape
    x = input

    # time axis @ 48 kHz
    t = torch.arange(N, device=x.device, dtype=x.dtype) / float(SR)

    # percent → fraction
    d_wow  = max(0.0, float(largeWobbleMax)) / 100.0
    d_flut = max(0.0, float(tinyWobbleMax))  / 100.0

    out = torch.empty_like(x)

    for ch in range(C):
        # instantaneous speed(t)
        wow  = d_wow  * torch.sin(2*math.pi*largeWobbleFreq * t)
        flut = d_flut * torch.sin(2*math.pi*tinyWobbleFreq  * t)
        speed = 1.0 + wow + flut
        speed = speed / speed.mean()              # keep duration ≈ constant

        # cumulative playhead (fractional indices into x[ch])
        p = torch.cumsum(speed, dim=0)

        # cubic interpolation inline (no external helper)
        # keep kernel fully in-bounds: need indices ip-1 .. ip+2
        p_clamped = torch.clamp(p, 1.0, float(N - 3) - 1e-6)
        ip = torch.floor(p_clamped).long()
        tau = (p_clamped - ip.to(p_clamped.dtype)).to(x.dtype)

        y0 = x[ch, ip - 1]
        y1 = x[ch, ip + 0]
        y2 = x[ch, ip + 1]
        y3 = x[ch, ip + 2]

        out[ch] = _cubic_hermite(y0, y1, y2, y3, tau)

    # tiny DC cleanup + safety headroom
    out = _dc_block(out, f_hp=20.0)
    peak = torch.max(torch.abs(out))
    if peak > 0.999:
        out = out * (0.999 / peak)

    return out
if __name__ == "__main__":
    main()

## AI assisted: Comments, formatting, path stuff