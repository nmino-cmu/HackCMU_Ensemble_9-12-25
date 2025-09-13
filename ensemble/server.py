# server.py
import io
import uuid
from typing import Dict, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response

# ---------- IMPORT YOUR FUNCTIONS HERE ----------
# Example signatures you might have:
#   def vinyl_warp(wav_bytes: bytes, amount: float) -> bytes: ...
#   def surface_damage(wav_bytes: bytes, amount: float) -> bytes: ...
#   def dropout(wav_bytes: bytes, amount: float) -> bytes: ...
#   def eq_attenuation(wav_bytes: bytes, db: float) -> bytes: ...
#   def stylus_resonance(wav_bytes: bytes, amt: float) -> bytes: ...
#
# For demo, I'll provide a simple "identity" pipeline using pydub,
# so you can see the wiring. Replace `apply_chain` with *your* functions.
from pydub import AudioSegment, effects


def apply_chain(
    wav_bytes: bytes,
    vinyl_warp: float,
    surface_damage: float,
    dropout_amt: float,
    eq_db: float,
    resonance: float,
    preview_ms: int = None,
) -> bytes:
    """
    Demo chain. Replace the body with calls to YOUR functions.
    This one just normalizes and applies a simple EQ-ish tilt for illustration.
    """
    audio = AudioSegment.from_file(io.BytesIO(wav_bytes))
    if preview_ms:
        audio = audio[:preview_ms]

    # DEMO transforms (replace with your own pipeline):
    # normalize
    audio = effects.normalize(audio)

    # crude "EQ attenuation" demo: reduce some high end if eq_db > 0
    # (real EQ would use a proper filter)
    if eq_db > 0:
        audio = audio.low_pass_filter(12000 - min(eq_db, 24) * 300)

    # crude "resonance" demo: add a slight comb filter feel by overlaying a delayed copy
    if resonance > 0:
        delay_ms = int(2 + resonance * 6)  # 2–8ms
        audio = audio.overlay(audio - 6, position=delay_ms)

    # NOTE: vinyl_warp/surface_damage/dropout_amt ignored in this demo.
    # In your code, call your functions here, in order, passing the values above.

    # Export to WAV bytes
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


# ---------- App + in-memory store ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store original uploads in memory: {file_id: (filename, bytes)}
STORE: Dict[str, Tuple[str, bytes]] = {}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    fid = str(uuid.uuid4())
    STORE[fid] = (file.filename, data)
    return {"file_id": fid, "filename": file.filename}


@app.post("/process")
async def process_audio(
    file_id: str = Form(...),
    filename: str = Form(...),

    # Sliders (ALL optional: send what you need)
    vinyl_warp: float = Form(0.0),
    surface_damage: float = Form(0.0),
    dropout: float = Form(0.0),
    eq: float = Form(0.0),
    resonance: float = Form(0.0),

    # Optional: shorter preview for faster "live" feeling (e.g., 15s)
    preview_ms: int = Form(15000),
):
    item = STORE.get(file_id)
    if not item:
        return JSONResponse({"error": "file not found"}, status_code=404)

    _, wav_bytes = item

    # Call YOUR function(s) here. For now, demo chain:
    out_bytes = apply_chain(
        wav_bytes,
        vinyl_warp=vinyl_warp,
        surface_damage=surface_damage,
        dropout_amt=dropout,
        eq_db=eq,
        resonance=resonance,
        preview_ms=preview_ms,
    )

    return StreamingResponse(io.BytesIO(out_bytes), media_type="audio/wav")


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
    item = STORE.get(file_id)
    if not item:
        return JSONResponse({"error": "file not found"}, status_code=404)
    _, wav_bytes = item

    # Re-run chain on FULL file (no preview_ms)
    out_bytes = apply_chain(
        wav_bytes,
        vinyl_warp=vinyl_warp,
        surface_damage=surface_damage,
        dropout_amt=dropout,
        eq_db=eq,
        resonance=resonance,
        preview_ms=None,
    )
    return Response(
        content=out_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename.rsplit(".",1)[0]}_processed.wav"'
        },
    )
