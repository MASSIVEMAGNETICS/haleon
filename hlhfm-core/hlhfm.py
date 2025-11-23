"""
HyperLiquid Holographic Fractal Memory (HLHFM) Implementation
A production-grade holographic memory system with Redis persistence.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import redis
from tenacity import retry, stop_after_attempt, wait_exponential
from pybreaker import CircuitBreaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HoloEntry:
    """Represents a holographic memory entry."""
    key: np.ndarray
    val: np.ndarray
    t: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key.tolist() if isinstance(self.key, np.ndarray) else self.key,
            "val": self.val.tolist() if isinstance(self.val, np.ndarray) else self.val,
            "t": self.t,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HoloEntry':
        """Create HoloEntry from dictionary."""
        return cls(
            key=np.array(data["key"]),
            val=np.array(data["val"]),
            t=data["t"],
            meta=data["meta"]
        )


class HyperLiquidHolographicFractalMemory:
    """Base holographic fractal memory system."""
    
    def __init__(self, dim: int = 512, decay_rate: float = 0.1):
        """
        Initialize HLHFM.
        
        Args:
            dim: Dimensionality of holographic space
            decay_rate: Memory decay rate over time
        """
        self.dim = dim
        self.decay_rate = decay_rate
        self.entries: List[HoloEntry] = []
        self.holo_trace = np.zeros((dim, dim), dtype=np.float32)
        logger.info(f"Initialized HLHFM with dim={dim}, decay_rate={decay_rate}")
    
    def write(self, text: str, audio: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write a memory entry.
        
        Args:
            text: Text description
            audio: Audio data as numpy array
            meta: Metadata dictionary
            
        Returns:
            Dictionary with write confirmation
        """
        # Create holographic key from text (simplified encoding)
        key = self._encode_text(text)
        
        # Create entry
        entry = HoloEntry(
            key=key,
            val=audio,
            t=time.time(),
            meta=meta
        )
        
        self.entries.append(entry)
        
        # Update holographic trace
        self._update_trace(key, audio)
        
        logger.info(f"Written entry: {meta.get('echo_id', 'unknown')}")
        return {
            "status": "success",
            "echo_id": meta.get("echo_id"),
            "t": entry.t
        }
    
    def query(self, cue_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query memory with text cue.
        
        Args:
            cue_text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of matching entries
        """
        cue_key = self._encode_text(cue_text)
        
        # Calculate similarities with decay
        current_time = time.time()
        scores = []
        
        for entry in self.entries:
            # Cosine similarity
            similarity = np.dot(cue_key, entry.key) / (
                np.linalg.norm(cue_key) * np.linalg.norm(entry.key) + 1e-8
            )
            
            # Apply temporal decay
            time_diff = current_time - entry.t
            decay = np.exp(-self.decay_rate * time_diff / 3600)  # decay per hour
            
            score = similarity * decay
            scores.append((score, entry))
        
        # Sort by score and return top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for score, entry in scores[:top_k]:
            results.append({
                "score": float(score),
                "meta": entry.meta,
                "t": entry.t,
                "audio_shape": entry.val.shape if isinstance(entry.val, np.ndarray) else None
            })
        
        logger.info(f"Query '{cue_text}' returned {len(results)} results")
        return results
    
    def save(self, filepath: str):
        """Save memory snapshot to file."""
        data = {
            "dim": self.dim,
            "decay_rate": self.decay_rate,
            "entries": [entry.to_dict() for entry in self.entries],
            "holo_trace": self.holo_trace.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved snapshot to {filepath}")
    
    def load(self, filepath: str):
        """Load memory snapshot from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.dim = data["dim"]
        self.decay_rate = data["decay_rate"]
        self.entries = [HoloEntry.from_dict(e) for e in data["entries"]]
        self.holo_trace = np.array(data["holo_trace"])
        
        logger.info(f"Loaded snapshot from {filepath}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to holographic key (simplified)."""
        # Simple hash-based encoding for demonstration
        # In production, use proper embeddings (BERT, etc.)
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        key = np.random.randn(self.dim).astype(np.float32)
        key = key / (np.linalg.norm(key) + 1e-8)
        return key
    
    def _update_trace(self, key: np.ndarray, val: np.ndarray):
        """Update holographic trace with new entry."""
        # Simplified trace update
        # In production, implement proper holographic encoding
        if len(val.shape) == 1 and val.shape[0] == self.dim:
            outer = np.outer(key, val)
            self.holo_trace += outer * 0.1  # Small update to prevent overflow


class RedisHLHFM(HyperLiquidHolographicFractalMemory):
    """Redis-backed HLHFM for production persistence and fault tolerance."""
    
    def __init__(self, redis_url: str, **kwargs):
        """
        Initialize Redis-backed HLHFM.
        
        Args:
            redis_url: Redis connection URL
            **kwargs: Additional args for base HLHFM
        """
        super().__init__(**kwargs)
        self.redis_url = redis_url
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._prefix = "hlhfm:"
        self.circuit_breaker = CircuitBreaker(fail_max=3, reset_timeout=60)
        
        # Load existing entries from Redis
        self._load_from_redis()
        
        logger.info(f"Initialized RedisHLHFM connected to {redis_url}")
    
    def write(self, text: str, audio: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Write with Redis persistence."""
        result = super().write(text, audio, meta)
        
        # Serialize and store in Redis
        entry = self.entries[-1]  # Get the just-added entry
        entry_data = entry.to_dict()
        
        try:
            self._write_to_redis(meta["echo_id"], entry_data)
        except redis.RedisError as e:
            logger.error(f"Redis write failed: {e}, continuing with in-memory only")
        
        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _write_to_redis(self, echo_id: str, entry_data: Dict[str, Any]):
        """Write entry to Redis with retry logic."""
        key = f"{self._prefix}{echo_id}"
        serialized = json.dumps(entry_data)
        self.redis.set(key, serialized)
    
    @circuit_breaker
    def query(self, cue_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query with Redis fallback."""
        try:
            # Try to ensure Redis is in sync
            self._sync_from_redis()
        except redis.RedisError as e:
            logger.warning(f"Redis sync failed during query: {e}, using in-memory data")
        
        return super().query(cue_text, top_k=top_k)
    
    def _load_from_redis(self):
        """Load all entries from Redis on startup."""
        try:
            keys = self.redis.keys(f"{self._prefix}*")
            logger.info(f"Loading {len(keys)} entries from Redis")
            
            for key in keys:
                data_str = self.redis.get(key)
                if data_str:
                    data = json.loads(data_str)
                    entry = HoloEntry.from_dict(data)
                    self.entries.append(entry)
                    
                    # Reconstruct trace
                    if len(entry.val.shape) == 1 and entry.val.shape[0] == self.dim:
                        self._update_trace(entry.key, entry.val)
            
            logger.info(f"Loaded {len(self.entries)} entries from Redis")
        except redis.RedisError as e:
            logger.warning(f"Could not load from Redis: {e}, starting with empty memory")
    
    def _sync_from_redis(self):
        """Sync in-memory state with Redis (incremental)."""
        # In production, implement proper sync logic
        # For now, this is a placeholder
        pass
    
    def auto_snapshot(self, interval: int = 3600):
        """
        Periodically save snapshots.
        
        Args:
            interval: Snapshot interval in seconds
        """
        while True:
            try:
                snapshot_name = f"hlhfm_snapshot_{int(time.time())}.json"
                self.save(snapshot_name)
                logger.info(f"Auto-snapshot created: {snapshot_name}")
            except Exception as e:
                logger.error(f"Auto-snapshot failed: {e}")
            
            time.sleep(interval)
