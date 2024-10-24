import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance

def create_collections():
    qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)
    try:
        qdrant_client.create_collection(
            collection_name="keywordColA",
            vectors_config=models.VectorParams(size=1024, distance=Distance.COSINE),
            shard_number=2,
            sharding_method=models.ShardingMethod.AUTO,
            hnsw_config=models.HnswConfigDiff(m=24, ef_construct=400, full_scan_threshold=100),
            on_disk_payload=False,
            timeout=30,
        )

        qdrant_client.create_collection(
            collection_name="summaryColA",
            vectors_config=models.VectorParams(size=1024, distance=Distance.COSINE),
            shard_number=2,
            sharding_method=models.ShardingMethod.AUTO,
            hnsw_config=models.HnswConfigDiff(m=24, ef_construct=400, full_scan_threshold=100),
            on_disk_payload=False,
            timeout=30,
        )

        qdrant_client.create_collection(
            collection_name="contentColA",
            vectors_config=models.VectorParams(size=1024, distance=Distance.COSINE),
            shard_number=4,
            sharding_method=models.ShardingMethod.AUTO,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=400, full_scan_threshold=100),
            on_disk_payload=False,
            timeout=30,
        )

        print("Collections created successfully.")
    except Exception as e:
        print(f"Error creating collections: {e}")


if __name__ == "__main__":
    create_collections()