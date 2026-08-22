from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name="msmarco",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def build_index(chunks, force_rebuild=False):
    # If force_rebuild is True, delete the old collection so fresh passages are indexed
    if force_rebuild and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        # Add chunks into Qdrant
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
    q_vec = embedding_model.encode(query).tolist()
    # Using updated qdrant method query_points
    response = client.query_points(
        collection_name="msmarco", 
        query=q_vec, 
        limit=k
    )
    return [hit.payload["text"] for hit in response.points]