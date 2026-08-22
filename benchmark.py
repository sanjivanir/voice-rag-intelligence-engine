import time
import numpy as np
from ingest import (
    get_msmarco_passages, 
    fixed_overlap_chunking, 
    sentence_chunking, 
    metadata_aware_chunking
)
from retriever import build_index, retrieve_top_k
from harness import run_rag_pipeline

def evaluate_strategy(strategy_name, chunks, test_queries):
    print(f"\n--- Benchmarking Strategy: {strategy_name} ---")
    print(f"Total Chunks Generated: {len(chunks)}")
    
    # Re-index Qdrant with current chunk strategy
    build_index(chunks)
    
    latencies = []
    for q in test_queries:
        start_time = time.time()
        _ = run_rag_pipeline(q)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        latencies.append(latency)
        print(f"  Query: '{q}' | Latency: {latency:.2f} ms")
        
    p50 = np.percentile(latencies, 50)
    p70 = np.percentile(latencies, 70)
    p100 = np.percentile(latencies, 100)
    
    print(f"  P50: {p50:.2f} ms | P70: {p70:.2f} ms | P100: {p100:.2f} ms")
    return {"strategy": strategy_name, "p50": p50, "p70": p70, "p100": p100, "chunk_count": len(chunks)}

if __name__ == "__main__":
    raw_passages = get_msmarco_passages(limit=50)
    test_queries = [
        "What is the function of the heart?",
        "How does electricity flow through a circuit?",
        "What causes gravity in space?",
        "What is photosynthesis?"
    ]
    
    # 1. Fixed Overlap Chunks
    fixed_chunks = fixed_overlap_chunking(raw_passages)
    r1 = evaluate_strategy("Fixed Overlap (120/30)", fixed_chunks, test_queries)
    
    # 2. Sentence Boundary Chunks
    sent_chunks = sentence_chunking(raw_passages)
    r2 = evaluate_strategy("Sentence Boundary Split", sent_chunks, test_queries)
    
    # 3. Metadata-Aware Chunks
    meta_chunks = metadata_aware_chunking(raw_passages)
    r3 = evaluate_strategy("Metadata-Aware Contextual", meta_chunks, test_queries)

    print("\n================ FINAL COMPARISON ================")
    print(f"{'Strategy':<25} | {'Chunks':<8} | {'P50 (ms)':<10} | {'P100 (ms)':<10}")
    print("-" * 60)
    for res in [r1, r2, r3]:
        print(f"{res['strategy']:<25} | {res['chunk_count']:<8} | {res['p50']:<10.2f} | {res['p100']:<10.2f}")