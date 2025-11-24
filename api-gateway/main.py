"""
API Gateway Service
Unified REST interface for HLHFM + Eidolon-Ω system.
"""

import os
import uuid
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HLHFM + Eidolon-Ω API Gateway",
    description="Unified API for holographic memory and music generation",
    version="1.0.0"
)

# Service URLs
HLHFM_URL = os.getenv("HLHFM_URL", "http://localhost:8000")
EIDOLON_URL = os.getenv("EIDOLON_URL", "http://localhost:8001")
PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL", "http://localhost:8002")
EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://localhost:8003")


class GenerateMusicRequest(BaseModel):
    """Request model for end-to-end music generation."""
    text_prompt: str = Field(..., description="Text description for music generation")
    use_memory: bool = Field(default=True, description="Use HLHFM memory augmentation")
    save_to_memory: bool = Field(default=True, description="Save result to HLHFM")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)

class GenerateMusicResponse(BaseModel):
    """Response model for music generation."""
    waveform: List[float]
    text_prompt: str
    fad_score: float
    clap_score: float
    memory_augmented: bool


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    async with httpx.AsyncClient() as client:
        services = {
            "hlhfm": HLHFM_URL,
            "eidolon": EIDOLON_URL,
            "preprocessor": PREPROCESSOR_URL,
            "evaluator": EVALUATOR_URL
        }
        
        statuses = {}
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/health", timeout=5.0)
                statuses[name] = response.json().get("status", "unknown")
            except Exception as e:
                statuses[name] = f"error: {str(e)}"
        
        all_healthy = all(s == "healthy" for s in statuses.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "api-gateway",
            "services": statuses
        }

@app.post("/generate_music", response_model=GenerateMusicResponse)
async def generate_music(req: GenerateMusicRequest):
    """
    End-to-end music generation with optional memory augmentation.
    
    Flow:
    1. Query HLHFM for similar memories (if use_memory=True)
    2. Generate music with Eidolon-Ω
    3. Evaluate quality
    4. Save to HLHFM (if save_to_memory=True)
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Step 1: Query HLHFM if enabled
            memory_context = None
            if req.use_memory:
                try:
                    hlhfm_response = await client.post(
                        f"{HLHFM_URL}/query",
                        json={"cue_text": req.text_prompt, "top_k": 3}
                    )
                    memory_data = hlhfm_response.json()
                    if memory_data.get("results"):
                        memory_context = memory_data["results"]
                        logger.info(f"Retrieved {len(memory_context)} memories from HLHFM")
                except Exception as e:
                    logger.warning(f"HLHFM query failed: {e}, continuing without memory")
            
            # Step 2: Generate music
            eidolon_response = await client.post(
                f"{EIDOLON_URL}/generate",
                json={
                    "text": req.text_prompt,
                    "temperature": req.temperature,
                    "max_length": 100
                }
            )
            generation_data = eidolon_response.json()
            waveform = generation_data["waveform"]
            
            # Step 3: Evaluate quality
            eval_response = await client.post(
                f"{EVALUATOR_URL}/evaluate",
                json={
                    "generated_audio": waveform,
                    "reference_text": req.text_prompt
                }
            )
            eval_data = eval_response.json()
            
            # Step 4: Save to HLHFM if enabled
            if req.save_to_memory:
                try:
                    await client.post(
                        f"{HLHFM_URL}/write",
                        json={
                            "text": req.text_prompt,
                            "audio_data": waveform[:512],  # Truncate for demo
                            "metadata": {
                                "echo_id": str(uuid.uuid4()),
                                "fad_score": eval_data["fad_score"],
                                "clap_score": eval_data["clap_score"]
                            }
                        }
                    )
                    logger.info("Saved generation to HLHFM")
                except Exception as e:
                    logger.warning(f"Failed to save to HLHFM: {e}")
            
            return GenerateMusicResponse(
                waveform=waveform,
                text_prompt=req.text_prompt,
                fad_score=eval_data["fad_score"],
                clap_score=eval_data["clap_score"],
                memory_augmented=memory_context is not None
            )
        
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during generation: {e}")
            raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system-wide statistics."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            hlhfm_stats = await client.get(f"{HLHFM_URL}/stats")
            return {
                "hlhfm": hlhfm_stats.json(),
                "system": "operational"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
