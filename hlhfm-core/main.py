"""
HLHFM Core Service - FastAPI REST API
Production-grade service for holographic memory operations.
"""

import os
import json
import uuid
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from hlhfm import RedisHLHFM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
hlhfm_entries_total = Gauge('hlhfm_entries_total', 'Total entries in HLHFM')
hlhfm_query_latency_ms = Histogram('hlhfm_query_latency_ms', 'Query latency in milliseconds')
hlhfm_write_latency_ms = Histogram('hlhfm_write_latency_ms', 'Write latency in milliseconds')
hlhfm_holo_trace_norm = Gauge('hlhfm_holo_trace_norm', 'Norm of holographic trace')
hlhfm_queries_total = Counter('hlhfm_queries_total', 'Total queries processed')
hlhfm_writes_total = Counter('hlhfm_writes_total', 'Total writes processed')

# Global HLHFM instance
hlhfm_instance = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global hlhfm_instance
    
    # Startup
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    dim = int(os.getenv("HLHFM_DIM", "512"))
    decay_rate = float(os.getenv("HLHFM_DECAY_RATE", "0.1"))
    
    logger.info(f"Starting HLHFM Core Service (dim={dim}, decay={decay_rate})")
    hlhfm_instance = RedisHLHFM(redis_url=redis_url, dim=dim, decay_rate=decay_rate)
    
    # Update metrics
    hlhfm_entries_total.set(len(hlhfm_instance.entries))
    if hasattr(hlhfm_instance, 'holo_trace'):
        trace_norm = float(np.linalg.norm(hlhfm_instance.holo_trace))
        hlhfm_holo_trace_norm.set(trace_norm)
    
    yield
    
    # Shutdown
    logger.info("Shutting down HLHFM Core Service")


app = FastAPI(
    title="HLHFM Core Service",
    description="HyperLiquid Holographic Fractal Memory API",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Request/Response models
class WriteRequest(BaseModel):
    """Request model for writing to HLHFM."""
    text: str = Field(..., description="Text description of the memory")
    audio_data: List[float] = Field(..., description="Audio data as list of floats")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class WriteResponse(BaseModel):
    """Response model for write operations."""
    status: str
    echo_id: str
    timestamp: float

class QueryRequest(BaseModel):
    """Request model for querying HLHFM."""
    cue_text: str = Field(..., description="Query text to search memories")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")

class QueryResponse(BaseModel):
    """Response model for query operations."""
    results: List[Dict[str, Any]]
    query_text: str
    count: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "hlhfm-core",
        "entries": len(hlhfm_instance.entries) if hlhfm_instance else 0
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/write", response_model=WriteResponse)
@limiter.limit("10/minute")
async def write_memory(request: Request, write_req: WriteRequest):
    """
    Write a new memory entry to HLHFM.
    
    Rate limit: 10 requests per minute
    """
    if not hlhfm_instance:
        raise HTTPException(status_code=503, detail="HLHFM not initialized")
    
    try:
        with hlhfm_write_latency_ms.time():
            # Convert audio data to numpy array
            audio = np.array(write_req.audio_data, dtype=np.float32)
            
            # Add echo_id if not present
            if "echo_id" not in write_req.metadata:
                write_req.metadata["echo_id"] = str(uuid.uuid4())
            
            # Write to HLHFM
            result = hlhfm_instance.write(
                text=write_req.text,
                audio=audio,
                meta=write_req.metadata
            )
            
            # Update metrics
            hlhfm_writes_total.inc()
            hlhfm_entries_total.set(len(hlhfm_instance.entries))
            
            return WriteResponse(
                status=result["status"],
                echo_id=result.get("echo_id", write_req.metadata["echo_id"]),
                timestamp=result["t"]
            )
    
    except Exception as e:
        logger.error(f"Write failed: {e}")
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_memory(request: Request, query_req: QueryRequest):
    """
    Query HLHFM with text cue.
    
    Rate limit: 20 requests per minute
    """
    if not hlhfm_instance:
        raise HTTPException(status_code=503, detail="HLHFM not initialized")
    
    try:
        with hlhfm_query_latency_ms.time():
            results = hlhfm_instance.query(
                cue_text=query_req.cue_text,
                top_k=query_req.top_k
            )
            
            response = QueryResponse(
                results=results,
                query_text=query_req.cue_text,
                count=len(results)
            )
            
            # Update metrics only on success
            hlhfm_queries_total.inc()
            
            return response
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get HLHFM statistics."""
    if not hlhfm_instance:
        raise HTTPException(status_code=503, detail="HLHFM not initialized")
    
    trace_norm = float(np.linalg.norm(hlhfm_instance.holo_trace))
    
    return {
        "total_entries": len(hlhfm_instance.entries),
        "dim": hlhfm_instance.dim,
        "decay_rate": hlhfm_instance.decay_rate,
        "holo_trace_norm": trace_norm
    }

@app.post("/erase")
async def erase_user_data(user_id: str):
    """
    GDPR compliance: Erase user data.
    
    Args:
        user_id: User identifier to erase data for
    """
    if not hlhfm_instance:
        raise HTTPException(status_code=503, detail="HLHFM not initialized")
    
    try:
        # Find and remove entries with matching user_id
        removed_count = 0
        hlhfm_instance.entries = [
            entry for entry in hlhfm_instance.entries
            if entry.meta.get("user_id") != user_id
        ]
        
        # Remove from Redis
        if hasattr(hlhfm_instance, 'redis'):
            keys = hlhfm_instance.redis.keys(f"{hlhfm_instance._prefix}*")
            for key in keys:
                data_bytes = hlhfm_instance.redis.get(key)
                if data_bytes:
                    data = json.loads(data_bytes.decode('utf-8'))
                    if data.get("meta", {}).get("user_id") == user_id:
                        hlhfm_instance.redis.delete(key)
                        removed_count += 1
        
        # Update metrics
        hlhfm_entries_total.set(len(hlhfm_instance.entries))
        
        logger.info(f"Erased {removed_count} entries for user {user_id}")
        return {
            "status": "success",
            "user_id": user_id,
            "removed_entries": removed_count
        }
    
    except Exception as e:
        logger.error(f"Erase failed: {e}")
        raise HTTPException(status_code=500, detail=f"Erase failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
