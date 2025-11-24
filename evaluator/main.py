"""
Evaluator Service
Real-time FAD and CLAP scoring for generated music.
"""

import os
import logging
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evaluator Service",
    description="Music Quality Evaluation API (FAD/CLAP)",
    version="1.0.0"
)


class EvaluateRequest(BaseModel):
    """Request model for evaluation."""
    generated_audio: List[float] = Field(..., description="Generated audio waveform")
    reference_text: str = Field(..., description="Original text prompt")

class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    fad_score: float = Field(..., description="Frechet Audio Distance (lower is better)")
    clap_score: float = Field(..., description="CLAP similarity score (0-1, higher is better)")
    quality_metrics: dict


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "evaluator"}

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_audio(req: EvaluateRequest):
    """
    Evaluate generated audio quality using FAD and CLAP.
    
    FAD (Frechet Audio Distance): Measures distribution similarity
    CLAP (Contrastive Language-Audio Pretraining): Text-audio alignment
    
    Args:
        req: Evaluation request with audio and text
        
    Returns:
        Quality scores
    """
    try:
        # In production, use real FAD and CLAP models
        # For now, compute mock scores based on audio statistics
        
        audio_array = np.array(req.generated_audio)
        
        # Mock FAD score (lower is better, range ~0-100)
        # Based on audio variance and energy
        energy = np.mean(audio_array ** 2)
        variance = np.var(audio_array)
        fad_score = float(20.0 + 30.0 * (1.0 - min(energy, 1.0)))
        
        # Mock CLAP score (higher is better, range 0-1)
        # Based on audio length and text length correlation
        text_words = len(req.reference_text.split())
        audio_length = len(audio_array)
        expected_length = text_words * 5000  # ~0.2s per word
        
        if expected_length > 0:
            length_match = 1.0 - min(abs(audio_length - expected_length) / expected_length, 1.0)
        else:
            length_match = 0.5  # Default for empty text
        
        clap_score = float(0.5 + 0.4 * length_match)
        
        # Additional quality metrics
        quality_metrics = {
            "signal_to_noise_ratio": float(10.0 * np.log10(energy / (variance + 1e-8))),
            "dynamic_range": float(np.max(audio_array) - np.min(audio_array)),
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(audio_array)))) / 2.0),
            "audio_length_seconds": float(audio_length / 22050)
        }
        
        logger.info(f"Evaluated audio: FAD={fad_score:.2f}, CLAP={clap_score:.2f}")
        
        return EvaluateResponse(
            fad_score=fad_score,
            clap_score=clap_score,
            quality_metrics=quality_metrics
        )
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
