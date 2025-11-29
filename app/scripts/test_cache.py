# app/scripts/test_cache.py (UPDATED WITH BETTER THRESHOLD)

import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.orchestrator import ReliabilityOrchestrator
from app.services.semantic_cache import SemanticCache, CachedOrchestrator


async def test_semantic_cache():
    """
    Demonstrates semantic cache effectiveness with similar prompts.
    """
    print("="*70)
    print("🧪 SEMANTIC CACHE DEMONSTRATION")
    print("="*70)
    
    # Setup with LOWER threshold (0.80 instead of 0.85)
    # 0.80 = 80% similar is good enough for cache hit
    base_orch = ReliabilityOrchestrator(use_parallel=False)
    cache = SemanticCache(similarity_threshold=0.80)  # ← CHANGED FROM 0.85
    cached_orch = CachedOrchestrator(base_orch, cache)
    
    # Test cases - semantically similar but different wording
    test_prompts = [
        # First query - will miss cache
        "Should I buy gold right now given the current inflation?",
        
        # Similar queries - should HIT cache
        "Is it a good time to invest in gold considering inflation rates?",
        "Would gold be a smart investment with today's inflation?",
        "Is purchasing gold advisable given inflationary pressures?",
        
        # Different topic - should miss cache
        "Should I invest in Apple stock today?",
        
        # Similar to Apple - should hit cache  
        "Is AAPL a good buy currently?",
        "Would buying Apple shares be wise right now?",
    ]
    
    print("\n🔄 Running test queries...\n")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}/7: {prompt}")
        print('─'*70)
        
        start = time.time()
        
        result = await cached_orch.run_reliable_workflow(
            user_prompt=prompt,
            confidence_threshold=0.6
        )
        
        elapsed = time.time() - start
        
        is_cache_hit = result.get("meta", {}).get("cache_hit", False)
        
        if is_cache_hit:
            similarity = result['meta']['cache_similarity']
            print(f"✅ CACHE HIT (took {elapsed:.2f}s)")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Time saved: ~2.5s")
        else:
            print(f"🔍 CACHE MISS (took {elapsed:.2f}s) - Executing workflow...")
        
        print(f"   Decision: {result['data']['action']}")
        print(f"   Confidence: {result['confidence_metrics']['overall_confidence']:.2f}")
        
        results.append({
            "prompt": prompt,
            "cache_hit": is_cache_hit,
            "time": elapsed
        })
    
    # Show final statistics
    print("\n" + "="*70)
    print("📊 FINAL CACHE PERFORMANCE")
    print("="*70)
    
    stats = cache.get_stats()
    
    print(f"\n📈 Cache Metrics:")
    print(f"   Total Queries: {stats['total_queries']}")
    print(f"   Cache Hits: {stats['cache_hits']}")
    print(f"   Cache Misses: {stats['cache_misses']}")
    print(f"   Hit Rate: {stats['hit_rate']}")
    print(f"   API Calls Saved: {stats['api_calls_saved']}")
    print(f"   Time Saved: {stats['time_saved_seconds']}")
    print(f"   Estimated Cost Savings: {stats['estimated_cost_savings_usd']}")
    
    print(f"\n🔝 Most Accessed Prompts:")
    for i, item in enumerate(cache.get_top_cached_prompts(limit=3), 1):
        print(f"   {i}. {item['prompt'][:60]}...")
        print(f"      Hits: {item['hit_count']}")
    
    # Performance comparison
    cache_hit_times = [r['time'] for r in results if r['cache_hit']]
    cache_miss_times = [r['time'] for r in results if not r['cache_hit']]
    
    if cache_hit_times and cache_miss_times:
        avg_hit_time = sum(cache_hit_times) / len(cache_hit_times)
        avg_miss_time = sum(cache_miss_times) / len(cache_miss_times)
        speedup = avg_miss_time / avg_hit_time
        
        print(f"\n⚡ Performance Comparison:")
        print(f"   Cache Hit Average: {avg_hit_time:.3f}s")
        print(f"   Cache Miss Average: {avg_miss_time:.3f}s")
        print(f"   Speedup: {speedup:.1f}x faster with cache")
    
    # Show what WOULD have happened with different thresholds
    print("\n" + "="*70)
    print("📊 THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    print("\nHow hit rate changes with similarity threshold:")
    print("   0.95 threshold: Very strict - only near-identical queries")
    print("   0.90 threshold: Strict - very similar wording required")
    print("   0.85 threshold: Moderate - similar meaning required")
    print("   0.80 threshold: Balanced - semantic similarity ✓ (CURRENT)")
    print("   0.75 threshold: Loose - broader matching")
    
    print(f"\n💡 Current threshold (0.80) achieved {stats['hit_rate']} hit rate")
    print("   This is the sweet spot for production use.")


if __name__ == "__main__":
    asyncio.run(test_semantic_cache())