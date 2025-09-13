import torchaudio
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import math


WINDOW_DURATION_MS = 23
STEP_PERCENTAGE = 0.3
ATTENUATION_AMOUNT = 6.0

class audiofile:
    def __init__(self):
        self.sample_rate = 44100
        self.tensor = load_tensor_from_outputs()
    
    def get_tensor(self):
        return self.tensor

    def get_sample_rate(self):
        return self.sample_rate
    
    
 

def repo_root_from_here() -> Path:
    """
    py/dsp/eq/eq_tilt.py
    So rep root is three up.
    """
    return Path(__file__).resolve().parents[3]


def load_tensor_from_outputs() -> torch.Tensor:
    folder = repo_root_from_here() / "sound_data" / "warps"
    pt_files = list(folder.glob("*.pt"))
    
    tensor = torch.load(str(pt_files[0]), map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor
    
    raise TypeError(f"Loaded object is {type(tensor)}; not a tensor.")


def eq_fourier(audiofile):
    window_size = round(WINDOW_DURATION_MS/1000 * audiofile.get_sample_rate())
    step_size = int(STEP_PERCENTAGE * window_size)
    return torch.stft(
        input=audiofile.get_tensor(),
        n_fft=window_size,
        hop_lengh=step_size,
        window=torch.hann_window(window_size),
        return_complex=True
    )

def eq_inverse_fourier(fourier_tensor):
    window_size = round(WINDOW_DURATION_MS/1000 * audiofile.get_sample_rate())
    step_size = int(STEP_PERCENTAGE * window_size)
    return torch.istft(
        input=fourier_tensor,
        n_fft=window_size,
        hop_lengh=step_size,
        window=torch.hann_window(window_size),
        length=fourier_tensor.numel()
    )

# Measures the audiofile tensor and returns a float (0-1) representing how warm a file is (strong 150 Hz – 600 Hz, weak >2 kHz)
def how_warm(audiofile):
    return 0

def sigmoid(input):
    return 1 / (1 + math.e ** (-1 * input))

def standard_deviation(input_tensor):
    return torch.std(input_tensor, correction=0)

def mean_of_tensor(input_tensor):
    return torch.mean(input_tensor)

x = audiofile()
mag = x.get_tensor().abs()
print(mag)



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




# Measures the audiofile tensor and returns a float (0-1) representing how warm a file is (strong 150 Hz – 600 Hz, weak >2 kHz)
def how_warm(audiofile):
    return 0

def sigmoid(input):
    return 1 / (1 + math.e ** (-1 * input))

def standard_deviation(input_tensor):
    return torch.std(input_tensor, correction=0)

def mean_of_tensor(input_tensor):
    return torch.mean(input_tensor)

def reduce_highs_stft(
    x: torch.Tensor,
    sr: int,
    shelf_db: float,      # target attenuation at "far high" (linear above cutoff)
    cutoff_hz: float = 2000.0,   # start reducing above this
    transition_bw: float = 800.0,# smooth transition width (Hz)
    win_ms: float = 23.0,
    hop_pct: float = 0.25
) -> torch.Tensor:
    """
    Tensor in -> tensor out. Reduces >cutoff_hz content with a smooth high shelf in the STFT domain.
    x: [T] or [C,T] (float/real). Returns same shape.
    """
    # --- shape to [C,T] ---
    mono = False
    if x.ndim == 1:
        x = x.unsqueeze(0)   # [1,T]
        mono = True
    x = x.float()

    # --- STFT params ---
    win = max(16, int(round(win_ms/1000.0 * sr)))
    hop = max(1, int(round(hop_pct * win)))
    n_fft = 1 << math.ceil(math.log2(max(16, win)))
    window = torch.hann_window(win, device=x.device)

    # --- STFT (batched over channels) ---
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True
    )                        # [C, F, N]

    # --- Build frequency-dependent gain (smooth high shelf) ---
    F = X.size(-2)
    freqs = torch.linspace(0.0, sr/2, F, device=X.device).view(1, F, 1)   # [1,F,1]

    # Linear ramp from 0 dB at cutoff - bw  →  shelf_db at high end
    # w in [0,1] above cutoff; smooth around the boundary using transition_bw
    if transition_bw <= 0:
        w = (freqs >= cutoff_hz).float()
    else:
        w = torch.clamp((freqs - (cutoff_hz - transition_bw)) / (transition_bw), 0.0, 1.0)

    gain_lin = 1.0 + (10.0**(shelf_db/20.0) - 1.0) * w        # [1,F,1]
    Y = X * gain_lin                                           # apply shelf

    # --- Inverse STFT (batched) ---
    y = torch.istft(
        Y, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, length=x.shape[-1]
    )                        # [C,T]

    return y.squeeze(0) if mono else y

file = audiofile()
fourier_file = eq_fourier(file)
fourier_file = reduce_highs_stft(fourier_file, 44100, ATTENUATION_AMOUNT)
newfile = eq_inverse_fourier(fourier_file)
_save_wav_pcm24(newfile, repo_root_from_here() / "sound_data" / "eq")

# reduce_high_stft was written with help from AI