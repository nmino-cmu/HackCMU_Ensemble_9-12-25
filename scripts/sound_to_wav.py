import ffmpeg
import os
from pathlib import Path



def main():
    HERE = Path(__file__).resolve().parent
    SRC  = HERE.parent / "sound_data" / "data_raw"

    for p in SRC.iterdir():
        if p.is_file():
            print("Input:", p.resolve())

        
    input_folder = "data_raw"
    output_folder = "data_wav"
    # abs_path = os.path.abspath(Path);

    (
    ffmpeg.input()
    .output('output.wav', acodec="pcm_s32le", ar=96000, ac = 1)
    .run()
    )


if __name__ == "__main__":
    main()