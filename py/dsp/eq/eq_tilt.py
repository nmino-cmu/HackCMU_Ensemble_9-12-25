import torchaudio
import torch
from dsp.audiofile import audiofile

WINDOW_DURATION_MS = 23
STEP_PERCENTAGE = 0.3

# Takes in an audiofile erforms a stft and returns the tensor output
def eq_fourier(audiofile):
    window_size = WINDOW_DURATION_MS * audiofile.sample_rate
    step_size = STEP_PERCENTAGE * window_size
    
    return torch.stft(
        input=audiofile.tensor,
        n_fft=window_size,
        hop_length=step_size,
        window=torch.hann_window(window_size),
        return_complex=True
    )
