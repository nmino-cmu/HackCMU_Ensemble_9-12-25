### Dependencies:
- Python 3.10–3.12 (with venv)
pip3 install pathlib
pip3 install scipy
pip3 install torch
pip3 install ffmpeg-python
pip3 install Torchaudio
pip3 isntall numpy
- [ffmpeg](https://ffmpeg.org/) (binary must be on PATH)
  - macOS: `brew install ffmpeg`
  - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
  - Windows (PowerShell + Chocolatey): `choco install ffmpeg`
  - Windows (manual): Download from https://www.gyan.dev/ffmpeg/builds/ and add `bin/` to PATH



Training Datasets:
Scratch Noises in Vinyl - Kaggle
https://www.kaggle.com/datasets/seandaly/detecting-scratch-noise-in-vinyl-playback
Filename Convention:
Section Number - Acronym of the Album - sect0/sect1 (clean/scratchy).wav

Clean music & Noise (just noise) - github
https://github.com/slliugit/slliugit.github.io
