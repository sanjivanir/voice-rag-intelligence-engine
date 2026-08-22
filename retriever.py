import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Global variables to retain memory across Streamlit reruns
_qdrant_client = None
_all_chunks_cache = []
COLLECTION_NAME = "msmarco_chunks"

def get_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
    return _qdrant_client

def build_index(chunks, force_rebuild=False):
    """Builds Qdrant vector index and caches passages locally."""
    global _all_chunks_cache
    _all_chunks_cache = chunks
    
    client = get_client()
    
    if force_rebuild and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        points = [
            PointStruct(
                id=idx,
                vector=encoder.encode(chunk).tolist(),
                payload={"text": chunk}
            )
            for idx, chunk in enumerate(chunks)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

def retrieve_top_k(query, k=2):
    """Hybrid Retrieval: Searches Qdrant vectors + Keyword fallback."""
    global _all_chunks_cache
    matched_contexts = []
    
    # 1. Direct Keyword Matching Fallback
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    for chunk in _all_chunks_cache:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in query_words):
            matched_contexts.append(chunk)
            if len(matched_contexts) >= k:
                return matched_contexts

    # 2. Vector Search Retrieval
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        query_vector = encoder.encode(query).tolist()
        
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=k
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=k
            )
            
        vector_matches = [hit.payload["text"] for hit in hits if "text" in hit.payload]
        if vector_matches:
            return vector_matches

    return matched_contexts