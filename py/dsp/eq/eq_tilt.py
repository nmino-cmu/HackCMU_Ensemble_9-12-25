import torchaudio
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import math


WINDOW_DURATION_MS = 23
STEP_PERCENTAGE = 0.3

class audiofile:
    def __init__(self):
        self.sample_rate = 24
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
    folder = repo_root_from_here() / "sound_data" / "outputs"
    pt_files = list(folder.glob("*.pt"))
    
    tensor = torch.load(str(pt_files[0]), map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor
    
    raise TypeError(f"Loaded object is {type(tensor)}; not a tensor.")



# Takes in an audiofile performs a stft and returns the tensor output
def eq_fourier(audiofile):
    window_size = WINDOW_DURATION_MS * audiofile.get_sample_rate()
    step_size = int(STEP_PERCENTAGE * window_size)
    
    return torch.stft(
        input=audiofile.get_tensor(),
        n_fft=window_size,
        hop_length=step_size,
        window=torch.hann_window(window_size),
        return_complex=True
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
mag = eq_fourier(x).abs()
# plt.imshow(.numpy(), aspect='auto', origin='lower')
# plt.colorbar()
# plt.show()
# print()
                             # [freq, time]
Mag_db = 20 * torch.log10(mag + 1e-10)      # avoid log(0)

import matplotlib.pyplot as plt
plt.imshow(Mag_db.cpu().numpy(), origin="lower", aspect="auto",
           vmin=-80, vmax=0)                # clamp to a useful dB range
plt.colorbar(); plt.xlabel("time frames"); plt.ylabel("freq bins")

# Dict to tensor conversion in load_tensor_from_outputs() was done with AI