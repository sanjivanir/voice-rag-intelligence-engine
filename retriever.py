import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize embedding model and Qdrant in-memory client
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")
COLLECTION_NAME = "msmarco_chunks"

def build_index(chunks, force_rebuild=False):
    """Builds or rebuilds the Qdrant vector collection."""
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
    """Searches Qdrant for top-k matching passage chunks."""
    if not client.collection_exists(COLLECTION_NAME):
        return []
        
    query_vector = encoder.encode(query).tolist()
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    
    return [hit.payload["text"] for hit in search_result if "text" in hit.payload]