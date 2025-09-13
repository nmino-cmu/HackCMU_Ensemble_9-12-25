import numpy
import scipy
import torchaudio
from pathlib import Path

largeWobbleMax = 3
tinyWobbleMax = 1
# +- percent plaback speed

largeWobbleFreq = 1
tinyWobbleFreq = 1
# hz



def main():
    warper(load_first_wav_as_tensor(), largeWobbleMax, largeWobbleFreq, tinyWobbleMax, tinyWobbleFreq)



    return

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

def warper(input, largeWobbleMax, largeWobbleFreq, tinyWobbleMax, tinyWobbleFreq):
    # input is the big ass tensor
    return

if __name__ == "__main__":
    main()