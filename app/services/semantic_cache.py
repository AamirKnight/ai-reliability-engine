
import numpy as np
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque
from google import genai
from app.core.config import settings

class CacheEntry:
    """Represents a single cached result"""
    def __init__(self, prompt: str, result: Dict[Any, Any], embedding: np.ndarray):
        self.prompt = prompt
        self.result = result
        self.embedding = embedding
        self.created_at = datetime.now()
        self.hit_count = 0
        self.last_accessed = datetime.now()


class SemanticCache:
    """
    Intelligent semantic caching using Google's Embedding API.
    Stateless and memory-efficient (no local PyTorch required).
    """
    
    def __init__(
        self, 
        similarity_threshold: float = 0.80,
        max_cache_size: int = 1000,
        ttl_hours: int = 24
    ):
        print("🔄 Initializing Semantic Cache (API-based)...")
        # Use Google GenAI Client instead of local heavy models
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "text-embedding-004"
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # Storage
        self.cache: deque[CacheEntry] = deque(maxlen=max_cache_size)
        
        # Metrics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved_seconds": 0.0,
            "api_calls_saved": 0,
            "embedding_errors": 0
        }
    
    def _encode_prompt(self, prompt: str) -> Optional[np.ndarray]:
        """
        Convert prompt to embedding vector using Google API.
        Returns None if API fails (failsafe).
        """
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=prompt
            )
            # Access the first embedding and convert to numpy array
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"⚠️ Embedding API Failed: {str(e)}")
            self.stats["embedding_errors"] += 1
            return None
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def _clean_expired_entries(self):
        """Remove entries older than TTL"""
        now = datetime.now()
        valid_entries = [
            entry for entry in self.cache 
            if (now - entry.created_at) < self.ttl
        ]
        self.cache = deque(valid_entries, maxlen=self.max_cache_size)
    
    def get(self, prompt: str, workflow_context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Retrieve cached result for semantically similar prompt.
        """
        self.stats["total_queries"] += 1
        
        if self.stats["total_queries"] % 100 == 0:
            self._clean_expired_entries()
        
        if not self.cache:
            self.stats["cache_misses"] += 1
            return None
        
        # Encode the query (API Call)
        query_embedding = self._encode_prompt(prompt)
        
        # If embedding failed (e.g. network issue), treat as cache miss
        if query_embedding is None:
            self.stats["cache_misses"] += 1
            return None
        
        # Find best match
        best_match: Optional[CacheEntry] = None
        best_similarity = 0.0
        
        for entry in self.cache:
            similarity = self._calculate_similarity(query_embedding, entry.embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            # Debug log
            pass

        # Check if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            # Cache hit!
            best_match.hit_count += 1
            best_match.last_accessed = datetime.now()
            
            self.stats["cache_hits"] += 1
            self.stats["api_calls_saved"] += 1
            self.stats["total_time_saved_seconds"] += 2.5
            
            return {
                "cache_hit": True,
                "similarity": best_similarity,
                "cached_prompt": best_match.prompt,
                "result": best_match.result,
                "cached_at": best_match.created_at.isoformat(),
                "hit_count": best_match.hit_count,
                "time_saved_seconds": 2.5
            }
        
        # Cache miss
        self.stats["cache_misses"] += 1
        return None
    
    def set(self, prompt: str, result: Dict[Any, Any], workflow_context: Optional[Dict] = None):
        """
        Store a new result in the cache.
        """
        # Encode the prompt
        embedding = self._encode_prompt(prompt)
        
        # If embedding failed, we just skip caching this item
        if embedding is None:
            return
        
        # Create cache entry
        entry = CacheEntry(
            prompt=prompt,
            result=result,
            embedding=embedding
        )
        
        self.cache.append(entry)
    
    def invalidate(self, prompt: str = None):
        """
        Invalidate cache entries.
        """
        if prompt is None:
            self.cache.clear()
            return
        
        query_embedding = self._encode_prompt(prompt)
        if query_embedding is None:
            return

        # Remove similar entries
        filtered = [
            entry for entry in self.cache
            if self._calculate_similarity(query_embedding, entry.embedding) < self.similarity_threshold
        ]
        
        self.cache = deque(filtered, maxlen=self.max_cache_size)
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_queries"]
        
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.max_cache_size,
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": f"{hit_rate * 100:.1f}%",
            "api_calls_saved": self.stats["api_calls_saved"],
            "embedding_errors": self.stats.get("embedding_errors", 0),
            "time_saved_seconds": f"{self.stats['total_time_saved_seconds']:.1f}s",
            "similarity_threshold": self.similarity_threshold
        }
    
    def get_top_cached_prompts(self, limit: int = 10) -> list:
        """Get most frequently accessed cached prompts"""
        sorted_cache = sorted(self.cache, key=lambda e: e.hit_count, reverse=True)
        
        return [
            {
                "prompt": entry.prompt[:100] + "..." if len(entry.prompt) > 100 else entry.prompt,
                "hit_count": entry.hit_count,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat()
            }
            for entry in sorted_cache[:limit]
        ]
    
    def export_cache(self, filepath: str):
        """Export cache to file for persistence"""
        cache_data = {
            "entries": [
                {
                    "prompt": entry.prompt,
                    "result": entry.result,
                    "embedding": entry.embedding.tolist(),
                    "created_at": entry.created_at.isoformat(),
                    "hit_count": entry.hit_count
                }
                for entry in self.cache
            ],
            "stats": self.stats,
            "config": {
                "similarity_threshold": self.similarity_threshold,
                "max_cache_size": self.max_cache_size,
                "ttl_hours": self.ttl.total_seconds() / 3600
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✅ Cache exported to {filepath}")

# ============================================================================
# INTEGRATION WRAPPER
# ============================================================================

class CachedOrchestrator:
    """
    Wrapper around ReliabilityOrchestrator that adds semantic caching.
    Drop-in replacement for the original orchestrator.
    """
    
    def __init__(self, orchestrator, cache: Optional[SemanticCache] = None):
        self.orchestrator = orchestrator
        self.cache = cache or SemanticCache(
            similarity_threshold=0.80, # Updated to match SemanticCache default
            max_cache_size=1000,
            ttl_hours=24
        )
    
    async def run_reliable_workflow(
        self,
        user_prompt: str,
        confidence_threshold: float = 0.7,
        response_schema = None,
        use_cache: bool = True
    ):
        """
        Execute workflow with semantic caching.
        
        Args:
            user_prompt: The prompt to execute
            confidence_threshold: Minimum confidence required
            response_schema: Pydantic schema for validation
            use_cache: Whether to use cache (disable for testing)
        """
        
        # Try cache first
        if use_cache:
            cached_result = self.cache.get(user_prompt)
            
            if cached_result:
                print(f"⚡ CACHE HIT! (similarity: {cached_result['similarity']:.3f})")
                print(f"   Saved ~2.5s and 1 API call")
                print(f"   Original prompt: {cached_result['cached_prompt'][:80]}...")
                
                # Return in same format as orchestrator
                return {
                    "status": "success",
                    "data": cached_result["result"]["data"],
                    "confidence_metrics": cached_result["result"]["confidence_metrics"],
                    "meta": {
                        **cached_result["result"]["meta"],
                        "cache_hit": True,
                        "cache_similarity": cached_result["similarity"],
                        "time_saved_seconds": cached_result["time_saved_seconds"]
                    }
                }
        
        # Cache miss - execute normally
        print("🔍 Cache miss - executing workflow...")
        
        result = await self.orchestrator.run_reliable_workflow(
            user_prompt=user_prompt,
            confidence_threshold=confidence_threshold,
            response_schema=response_schema
        )
        
        # Store successful results in cache
        if result["status"] == "success" and use_cache:
            self.cache.set(user_prompt, result)
            print("💾 Result cached for future queries")
        
        return result
    
    def health_check(self):
        """Pass through to orchestrator"""
        return self.orchestrator.health_check()
    
    def get_cache_stats(self):
        """Get cache performance metrics"""
        return self.cache.get_stats()
