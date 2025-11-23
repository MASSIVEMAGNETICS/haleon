"""
Preprocessor Service
Audio and text preprocessing for HLHFM and Eidolon-Ω.
"""

import os
import logging
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessor Service",
    description="Audio/Text Preprocessing API",
    version="1.0.0"
)


class TextPreprocessRequest(BaseModel):
    """Request model for text preprocessing."""
    text: str = Field(..., description="Input text to preprocess")

class TextPreprocessResponse(BaseModel):
    """Response model for text preprocessing."""
    tokens: List[int] = Field(..., description="Token IDs")
    token_count: int

class AudioPreprocessResponse(BaseModel):
    """Response model for audio preprocessing."""
    mel_spectrogram: List[List[float]] = Field(..., description="Mel-spectrogram")
    sample_rate: int
    duration: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "preprocessor"}

@app.post("/preprocess_text", response_model=TextPreprocessResponse)
async def preprocess_text(req: TextPreprocessRequest):
    """
    Preprocess text using BPE tokenization.
    
    Args:
        req: Text preprocessing request
        
    Returns:
        Tokenized text
    """
    try:
        # Simplified tokenization (in production, use SentencePiece)
        tokens = [hash(word) % 50000 for word in req.text.lower().split()]
        
        return TextPreprocessResponse(
            tokens=tokens,
            token_count=len(tokens)
        )
    
    except Exception as e:
        logger.error(f"Text preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.post("/preprocess_audio", response_model=AudioPreprocessResponse)
async def preprocess_audio(audio_file: UploadFile = File(...)):
    """
    Convert audio to mel-spectrogram.
    
    Args:
        audio_file: Audio file (WAV, MP3, etc.)
        
    Returns:
        Mel-spectrogram representation
    """
    import tempfile
    
    temp_path = None
    try:
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Save to secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)
        
        # Load with librosa
        y, sr = librosa.load(temp_path, sr=22050)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return AudioPreprocessResponse(
            mel_spectrogram=mel_spec_db.tolist(),
            sample_rate=sr,
            duration=float(len(y) / sr)
        )
    
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")
    
    finally:
        # Ensure cleanup even if exception occurs
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
