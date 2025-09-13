from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

# Accept common audio + video; extend as needed
EXTS = set([".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"])

CODECS = {
    16: "pcm_s16le",
    24: "pcm_s24le",
    # 32f = 32-bit float WAV (huge, but true “no headroom” lossless for processing)
    32: "pcm_f32le",
}

def find_ffmpeg_or_die() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit(2)
    return exe

def iter_media(src: Path) -> Iterable[Path]:
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p

def build_out_path(inp: Path, src_root: Path, dst_root: Path) -> Path:
    rel = inp.relative_to(src_root)
    return (dst_root / rel).with_suffix(".wav")

def build_ffmpeg_cmd(ffmpeg_exe: str, inp: Path, outp: Path, sr: int, bitdepth: int, channels: int | None, soxr: bool, loudnorm: bool) -> list[str]:
    acodec = CODECS[bitdepth]
    args = [
        ffmpeg_exe,
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y",  # overwrite controlled at our level; ffmpeg still needs -y if we decided to overwrite
        "-i", str(inp),
        "-vn",  # ignore video; extract audio
    ]
    if soxr:
        # high-quality resampler
        args += ["-af", "aresample=resampler=soxr"]
    if loudnorm:
        # EBU R128 loudness normalization (conservative defaults)
        args += ["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"]
    args += ["-ar", str(sr), "-acodec", acodec]
    if channels is not None:
        args += ["-ac", str(channels)]
    args += [str(outp)]
    return args

def convert_one(ffmpeg_exe: str, inp: Path, outp: Path, sr: int, bitdepth: int, channels: int | None, force: bool, dry_run: bool, soxr: bool, loudnorm: bool) -> tuple[Path, bool, str]:
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not force:
        return (inp, False, "skip (exists)")
    cmd = build_ffmpeg_cmd(ffmpeg_exe, inp, outp, sr, bitdepth, channels, soxr, loudnorm)
    if dry_run:
        return (inp, False, "dry-run")
    try:
        subprocess.run(cmd, check=True)
        return (inp, True, "ok")
    except subprocess.CalledProcessError as e:
        return (inp, False, f"FAIL (code {e.returncode})")

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_src = (here.parent / "sound_data" / "data_raw")
    default_dst = (here.parent / "sound_data" / "data_wav")

    ap = argparse.ArgumentParser(description="Bulk convert media to WAV")
    ap.add_argument("--input", "-i", type=Path, default=default_src, help="Input folder (default: sound_data/data_raw)")
    ap.add_argument("--output", "-o", type=Path, default=default_dst, help="Output folder (default: sound_data/data_wav)")
    ap.add_argument("--sr", type=int, default=48000, help="Sample rate (Hz), e.g., 44100 or 48000")
    ap.add_argument("--bitdepth", type=int, choices=CODECS.keys(), default=24, help="PCM bit depth: 16, 24, or 32 (float)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--channels", type=int, choices=[1, 2], help="Force 1=mono or 2=stereo")
    group.add_argument("--keep-channels", action="store_true", help="Preserve original channel count")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; don’t run ffmpeg")
    ap.add_argument("--soxr", action="store_true", help="Use soxr resampler (slower, higher quality)")
    ap.add_argument("--loudnorm", action="store_true", help="Apply EBU loudness normalization")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    ffmpeg_exe = find_ffmpeg_or_die()

    src: Path = args.input.resolve()
    dst: Path = args.output.resolve()
    if not src.exists():
        print(f"[ERR] Input folder not found: {src}", file=sys.stderr)
        return 2

    channels = None if args.keep_channels else (args.channels or 2)

    files = list(iter_media(src))
    if not files:
        print(f"[INFO] No media files found under: {src}")
        return 0

    print(f"[INFO] Converting {len(files)} file(s)")
    print(f"[INFO]  src: {src}")
    print(f"[INFO]  dst: {dst}")
    print(f"[INFO]  sr={args.sr} Hz, bitdepth={args.bitdepth}, channels={'preserve' if channels is None else channels}")
    if args.dry_run:
        print("[INFO] dry-run: no actual conversion will be performed")

    ok = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futures = []
        for inp in files:
            outp = build_out_path(inp, src, dst)
            futures.append(
                ex.submit(
                    convert_one, ffmpeg_exe, inp, outp,
                    args.sr, args.bitdepth, channels, args.force, args.dry_run,
                    args.soxr, args.loudnorm
                )
            )

        for fut in as_completed(futures):
            inp, changed, msg = fut.result()
            rel = inp.relative_to(src)
            print(f"[{msg:10}] {rel}")
            if msg.startswith("ok"):
                ok += 1
            elif msg.startswith("skip"):
                skipped += 1
            elif msg.startswith("dry"):
                skipped += 1
            else:
                failed += 1

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

# AI assisted: mostly human written











# import ffmpeg
# import os
# from pathlib import Path



# def main():
#     HERE = Path(__file__).resolve().parent
#     SRC  = HERE.parent / "sound_data" / "data_raw"

#     for p in SRC.iterdir():
#         if p.is_file():
#             print("Input:", p.resolve())

        
#     input_folder = "data_raw"
#     output_folder = "data_wav"
#     # abs_path = os.path.abspath(Path);

#     (
#     ffmpeg.input()
#     .output('output.wav', acodec="pcm_s32le", ar=96000, ac = 1)
#     .run()
#     )


# if __name__ == "__main__":
#     main()