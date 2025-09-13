import torch
import wave
import torchaudio
from torch import nn

class audiofile:
    def __init__(self, file_name):
        self.file_location = file_name
        self.tensor, self.sample_rate = torchaudio.load(self.file_location)
        

# reduces rolls off the very high frequencies (>15 kHz) and boosts low/mids (100-400Hz)
# takes the input of the 
def eq_measurement(music_file_location):
    vinylize_audio_file = audiofile(music_file_location)
    
