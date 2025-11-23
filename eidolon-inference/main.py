"""
Eidolon-Ω Inference Service
FastAPI service for music generation using ONNX Runtime.
"""

import os
import httpx
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from tenacity import retry, stop_after_attempt, wait_exponential

from eidolon_model import preprocess_text, postprocess_mel_spectrogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
eidolon_inference_latency_ms = Histogram('eidolon_inference_latency_ms', 'Inference latency in milliseconds')
eidolon_fad_score = Histogram('eidolon_fad_score', 'Frechet Audio Distance score')
eidolon_clap_score = Histogram('eidolon_clap_score', 'CLAP similarity score')
eidolon_generations_total = Counter('eidolon_generations_total', 'Total generations processed')

# Global ONNX session
onnx_session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global onnx_session
    
    # Startup
    model_path = os.getenv("ONNX_MODEL_PATH", "eidolon.onnx")
    
    if os.path.exists(model_path):
        logger.info(f"Loading ONNX model from {model_path}")
        onnx_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Add CUDAExecutionProvider for GPU
        )
        logger.info("ONNX model loaded successfully")
    else:
        logger.warning(f"ONNX model not found at {model_path}, will use mock generation")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Eidolon Inference Service")


app = FastAPI(
    title="Eidolon-Ω Inference Service",
    description="Text-to-Music Generation API",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for music generation."""
    text: str = Field(..., description="Text prompt for music generation")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling temperature")
    max_length: int = Field(default=100, ge=10, le=500, description="Maximum generation length")

class GenerateResponse(BaseModel):
    """Response model for music generation."""
    waveform: List[float] = Field(..., description="Generated audio waveform")
    text_prompt: str
    sample_rate: int = Field(default=22050, description="Audio sample rate")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "eidolon-inference",
        "model_loaded": onnx_session is not None
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@app.post("/generate", response_model=GenerateResponse)
async def generate_music(gen_req: GenerateRequest):
    """
    Generate music from text prompt.
    
    Retry policy: 3 attempts with exponential backoff
    """
    try:
        with eidolon_inference_latency_ms.time():
            if onnx_session:
                # Use ONNX model
                # Preprocess text to tokens
                text_tokens = preprocess_text(gen_req.text)
                text_tokens = text_tokens[:gen_req.max_length]  # Truncate if needed
                
                # Pad to batch
                text_tokens = np.expand_dims(text_tokens, axis=0).astype(np.int64)
                
                # Run inference
                outputs = onnx_session.run(
                    None,
                    {"text_tokens": text_tokens}
                )
                mel_spec = outputs[0]
                
                # Post-process to waveform
                waveform = postprocess_mel_spectrogram(mel_spec[0])
            else:
                # Mock generation for testing
                logger.warning("Using mock generation (ONNX model not loaded)")
                waveform = np.random.randn(22050 * 2).astype(np.float32)  # 2 seconds
                waveform = waveform / (np.abs(waveform).max() + 1e-8)
            
            # Update metrics
            eidolon_generations_total.inc()
            
            # Record quality scores (mock for development, in production use real evaluator)
            # TODO: Replace with real FAD/CLAP evaluation from evaluator service
            if os.getenv("MOCK_METRICS", "false").lower() == "true":
                eidolon_fad_score.observe(np.random.uniform(10, 50))
                eidolon_clap_score.observe(np.random.uniform(0.5, 0.9))
            
            return GenerateResponse(
                waveform=waveform.tolist(),
                text_prompt=gen_req.text,
                sample_rate=22050
            )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate_with_hlhfm")
async def generate_with_hlhfm(text: str, hlhfm_url: str = "http://hlhfm-core:8000"):
    """
    Generate music using HLHFM memory augmentation.
    
    Args:
        text: Text prompt
        hlhfm_url: URL of HLHFM service
    """
    try:
        # Query HLHFM for similar memories
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{hlhfm_url}/query",
                json={"cue_text": text, "top_k": 3}
            )
            hlhfm_results = response.json()
        
        # Augment generation with memory context
        augmented_text = text
        if hlhfm_results.get("results"):
            memory_context = " | ".join([
                r["meta"].get("text", "") for r in hlhfm_results["results"][:2]
            ])
            augmented_text = f"{text} (context: {memory_context})"
        
        # Generate with augmented text
        gen_req = GenerateRequest(text=augmented_text)
        return await generate_music(gen_req)
    
    except Exception as e:
        logger.error(f"HLHFM-augmented generation failed: {e}")
        # Fallback to regular generation
        gen_req = GenerateRequest(text=text)
        return await generate_music(gen_req)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
