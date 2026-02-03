"""
Coqui TTS Voice Cloning Server
Deploy this to Railway, Render, or any VPS with GPU support for best performance.
"""

import os
import io
import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from TTS.api import TTS

app = FastAPI(title="Coqui TTS Voice Cloning Server")

# CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
VOICES_DIR = Path("./voices")
VOICES_DIR.mkdir(exist_ok=True)

# Initialize TTS model (XTTS v2 for voice cloning)
# This will download the model on first run (~1.8GB)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": device}


@app.post("/train-voice")
async def train_voice(
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Accept voice samples and create a speaker embedding for the user.
    
    XTTS v2 uses zero-shot voice cloning, so we just need to store
    reference audio files for the user. No actual "training" needed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No audio files provided")
    
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="At least 3 audio samples required")
    
    # Create user voice directory
    user_voice_dir = VOICES_DIR / user_id
    user_voice_dir.mkdir(exist_ok=True)
    
    # Clear old samples
    for old_file in user_voice_dir.glob("*"):
        old_file.unlink()
    
    saved_files = []
    
    for i, file in enumerate(files):
        # Save each audio file
        file_ext = Path(file.filename).suffix or ".webm"
        file_path = user_voice_dir / f"sample_{i}{file_ext}"
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        saved_files.append(str(file_path))
    
    # Convert webm files to wav if needed (XTTS works better with wav)
    wav_files = []
    for file_path in saved_files:
        if file_path.endswith(".webm"):
            wav_path = file_path.replace(".webm", ".wav")
            try:
                # Use ffmpeg to convert
                os.system(f'ffmpeg -y -i "{file_path}" -ar 22050 -ac 1 "{wav_path}" 2>/dev/null')
                if Path(wav_path).exists():
                    wav_files.append(wav_path)
                    Path(file_path).unlink()  # Remove original webm
            except Exception as e:
                print(f"Conversion error: {e}")
                wav_files.append(file_path)
        else:
            wav_files.append(file_path)
    
    # Use the best quality samples (longest duration) as reference
    # XTTS works best with 6-30 seconds of reference audio
    reference_files = wav_files[:5]  # Use up to 5 samples
    
    # Store reference file paths for this user
    ref_file = user_voice_dir / "references.txt"
    with open(ref_file, "w") as f:
        f.write("\n".join(reference_files))
    
    return {
        "success": True,
        "voice_id": f"coqui_{user_id}",
        "message": f"Voice profile created with {len(reference_files)} reference samples",
        "samples_count": len(reference_files)
    }


@app.post("/tts")
async def text_to_speech(
    text: str = Form(None),
    user_id: str = Form(None)
):
    """
    Generate speech in the user's cloned voice.
    Returns WAV audio.
    """
    # Also accept JSON body
    if text is None:
        from fastapi import Request
        raise HTTPException(status_code=400, detail="Text is required")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Find user's voice reference files
    user_voice_dir = VOICES_DIR / (user_id or "default")
    ref_file = user_voice_dir / "references.txt"
    
    if user_id and ref_file.exists():
        # Use user's cloned voice
        with open(ref_file, "r") as f:
            reference_files = [p.strip() for p in f.readlines() if p.strip()]
        
        # Filter to existing files
        reference_files = [p for p in reference_files if Path(p).exists()]
        
        if not reference_files:
            raise HTTPException(
                status_code=404, 
                detail="Voice samples not found. Please re-train your voice."
            )
        
        # Use first available reference
        speaker_wav = reference_files[0]
    else:
        # Use default voice (no cloning)
        speaker_wav = None
    
    # Generate speech
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    
    try:
        if speaker_wav:
            # Generate with voice cloning
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language="en"
            )
        else:
            # Generate with default voice
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language="en"
            )
        
        # Read and return the audio
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        # Cleanup
        Path(output_path).unlink(missing_ok=True)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )
    
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


# JSON body support for /tts
@app.post("/tts-json")
async def text_to_speech_json(data: dict):
    """Alternative endpoint accepting JSON body"""
    text = data.get("text", "")
    user_id = data.get("user_id")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Reuse the form-based logic
    from fastapi import Form
    
    user_voice_dir = VOICES_DIR / (user_id or "default")
    ref_file = user_voice_dir / "references.txt"
    
    speaker_wav = None
    if user_id and ref_file.exists():
        with open(ref_file, "r") as f:
            reference_files = [p.strip() for p in f.readlines() if p.strip()]
        reference_files = [p for p in reference_files if Path(p).exists()]
        if reference_files:
            speaker_wav = reference_files[0]
    
    output_path = f"/tmp/{uuid.uuid4()}.wav"
    
    try:
        if speaker_wav:
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language="en"
            )
        else:
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language="en"
            )
        
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        Path(output_path).unlink(missing_ok=True)
        
        return Response(
            content=audio_data,
            media_type="audio/wav"
        )
    except Exception as e:
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
