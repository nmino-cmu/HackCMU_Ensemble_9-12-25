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
    print(load_first_wav_as_tensor())



    return

# input is a wav

def load_first_wav_as_tensor():
    """
    Load the first .wav file from sound_data/data_wav.
    Returns:
        waveform: torch.Tensor [C, N], float32 in [-1, 1]
        sr: int (sample rate, e.g. 48000)
    """
    folder = Path(__file__).resolve().parent.parent.parent / "sound_data" / "data_wav"
    wav_files = list(folder.glob("*.wav"))

    file_path = wav_files[0]
    waveform, sr = torchaudio.load(str(file_path))
    return waveform, sr

def warper(input, largeWobbleMax, largeWobbleFreq, tinyWobbleMax, tinyWobbleFreq):
    waveform, sr = torchaudio.load("your_file.wav")

if __name__ == "__main__":
    main()