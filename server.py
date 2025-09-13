import io, uuid
from typing import Dict, Tuple, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.responses import HTMLResponse
from pathlib import Path

import torch, torchaudio

from eq_tilt import reduce_highs_stft
from resonance_adder import stylus_resonance_fast
import vinyl_dropout_intensity as vdrop           
from vinyl_surface_damage import add_vinyl_crackle
from vinyl_warp import warper

SR = 48000

def _load(wav_bytes: bytes, target_sr=SR):
  x, sr = torchaudio.load(io.BytesIO(wav_bytes))
  if sr != target_sr:
    x = torchaudio.functional.resample(x, sr, target_sr)
  return x.float()

def _save(x: torch.Tensor, sr=SR) -> bytes:
  buf = io.BytesIO()
  torchaudio.save(buf, x.clamp(-1,1).cpu(), sr, format="wav", encoding="PCM_S", bits_per_sample=16)
  buf.seek(0); return buf.read()

def apply_chain(x: torch.Tensor, vinyl_warp_amt, surface_damage_amt, dropout_amt, eq_db, resonance_amt):
  y = x

  # 1) Wow/Flutter (keep your mapping; audible at higher values)
  if vinyl_warp_amt > 0:
    y = warper(y, vinyl_warp_amt / 50 * 0.6, vinyl_warp_amt / 50 * 0.8, vinyl_warp_amt / 50 * 0.2, vinyl_warp_amt / 50 * 12.0)

  # 2) Surface crackle — 0..100 slider → rho 0..1
  if surface_damage_amt > 0:
    rho = max(0.0, min(1.0, surface_damage_amt / 100.0))
    y = add_vinyl_crackle(y, rho=rho)

  # 3) Dropouts — set module INTENSITY directly (0..0.25), then apply once
  if dropout_amt > 0:
    vdrop.INTENSITY = 0 * float(dropout_amt) / 100.0 * 0.25   # 0..25% chance per block
    y = add_vinyl_crackle(y, 0.2)                        # use the module function

  # 4) EQ tilt — UI positive = more treble cut (negative shelf dB)
  if eq_db < 0:
    y = reduce_highs_stft(
      y, SR,
      shelf_db=eq_db, cutoff_hz=2000.0, transition_bw=800.0,
      win_ms=23.0, hop_pct=0.30
    )

  # 5) Stylus resonance — 0..1
  if resonance_amt > 0:
    y = stylus_resonance_fast(y, amount=resonance_amt)

  # Safety headroom
  peak = torch.max(torch.abs(y))
  if peak > 0.999:
    y = y * (0.999 / float(peak))
  return y


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STORE: Dict[str, Tuple[str, bytes]] = {}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
  data = await file.read()
  fid = str(uuid.uuid4())
  STORE[fid] = (file.filename, data)
  return {"file_id": fid, "filename": file.filename}

def _process_common(file_id: str, preview_ms: Optional[int], vinyl_warp: float, surface_damage: float, dropout: float, eq: float, resonance: float) -> bytes:
  if file_id not in STORE: raise FileNotFoundError
  filename, wav_bytes = STORE[file_id]
  x = _load(wav_bytes, SR)
  if preview_ms:
    n = int(SR * (preview_ms/1000.0))
    if x.shape[1] > n: x = x[:, :n].contiguous()
  y = apply_chain(x, vinyl_warp, surface_damage, dropout, eq, resonance)
  return _save(y, SR)

@app.post("/process")
async def process_audio(
  file_id: str = Form(...),
  filename: str = Form(...),
  vinyl_warp: float = Form(0.0),
  surface_damage: float = Form(0.0),
  dropout: float = Form(0.0),
  eq: float = Form(0.0),
  resonance: float = Form(0.0),
  preview_ms: int = Form(15000),
):
  try:
    print(f"[process] id={file_id[:8]} vw={vinyl_warp} sd={surface_damage} dr={dropout} eq={eq} rs={resonance} preview_ms={preview_ms}")
    out = _process_common(file_id, preview_ms, vinyl_warp, surface_damage, dropout, eq, resonance)
    return StreamingResponse(io.BytesIO(out), media_type="audio/wav")
  except FileNotFoundError:
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
def root():
    return Path("monofile.html").read_text(encoding="utf-8")


@app.post("/export")
async def export_audio(
  file_id: str = Form(...),
  filename: str = Form(...),
  vinyl_warp: float = Form(0.0),
  surface_damage: float = Form(0.0),
  dropout: float = Form(0.0),
  eq: float = Form(0.0),
  resonance: float = Form(0.0),
):
  try:
    out = _process_common(file_id, None, vinyl_warp, surface_damage, dropout, eq, resonance)
    base = filename.rsplit(".",1)[0] if "." in filename else filename
    return Response(content=out, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="{base}_processed.wav"'})
  except FileNotFoundError:
    return JSONResponse({"error": "file not found"}, status_code=404)
