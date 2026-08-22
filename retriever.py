import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "msmarco_chunks"

def retrieve_top_k_from_chunks(chunks, query, k=2):
    """Retrieves top matches directly from the active chunks list."""
    if not chunks:
        return []

    query_words = [w.lower() for w in query.split() if len(w) > 2]
    matched_contexts = []

    # 1. Direct Keyword Matching
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in query_words):
            matched_contexts.append(chunk)
            if len(matched_contexts) >= k:
                return matched_contexts

    # 2. Vector Search Fallback
    try:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=idx,
                vector=encoder.encode(c).tolist(),
                payload={"text": c}
            )
            for idx, c in enumerate(chunks)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

        query_vector = encoder.encode(query).tolist()
        if hasattr(client, "query_points"):
            response = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=k)
            hits = response.points
        else:
            hits = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=k)

        return [hit.payload["text"] for hit in hits if "text" in hit.payload]
    except Exception:
        return matched_contexts