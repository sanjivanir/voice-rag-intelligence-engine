from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(":memory:")

client.recreate_collection(
    collection_name="msmarco",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def build_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    points = [
        PointStruct(id=idx, vector=emb.tolist(), payload={"text": chunk})
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name="msmarco", points=points)

def retrieve_top_k(query, k=2):
    q_vec = embedding_model.encode(query).tolist()
    # Using updated qdrant method query_points
    response = client.query_points(
        collection_name="msmarco", 
        query=q_vec, 
        limit=k
    )
    return [hit.payload["text"] for hit in response.points]