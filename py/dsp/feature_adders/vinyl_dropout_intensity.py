import numpy
import scipy
import random
import torch
import torchaudio
import soundfile as sf
from pathlib import Path

INTENSITY: float = .01
BLOCK_MS : int = 0.2

def block_dropout(tensor, sr):
    C, N = tensor.shape
    block_len = int(sr * BLOCK_MS / 1000)
    dropped = tensor.clone()
    
    num_blocks = N // block_len
    
    for i in range(num_blocks):
        if torch.rand(1).item() < INTENSITY:
            start = i * block_len
            end = min((i+1) * block_len, N)
            dropped[:, start:end] = 0.0  
            
    fade_len = int(sr * 2 / 1000) 
    for ch in range(C):
        dropped[ch, start:start+fade_len] *= torch.linspace(1, 0, fade_len)
        dropped[ch, end-fade_len:end] *= torch.linspace(0, 1, fade_len)
    
    return dropped

def repo_root_from_here() -> Path:
    # file lives at py/dsp/eq/eq_tilt.py → repo root is three up
    return Path(__file__).resolve().parents[3]

def save_wav_pcm24(x: torch.Tensor, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y = x.detach().cpu().float().clamp_(-1.0, 1.0)
    sf.write(str(out_path), y.transpose(0, 1).numpy(), sr, subtype="PCM_24")

def _save_tensor(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x.detach().cpu(), str(path))

def main():
    input = repo_root_from_here() / "sound_data" / "outputs" / "eqed" / "eqed.wav"
    out_pt = repo_root_from_here() / "sound_data" / "outputs" / "dropouts" / "dropout.pt"
    out_wav = repo_root_from_here() / "sound_data" / "outputs" / "dropouts" / "dropout.wav"
    tensor, sr = torchaudio.load(input)
    _save_tensor(tensor, out_pt)
    dropped = block_dropout(tensor, sr)
    save_wav_pcm24(dropped, sr, out_wav)
    

if __name__ == "__main__":
    main()