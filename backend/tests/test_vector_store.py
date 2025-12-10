"""
Test script for VectorStore initialization and basic operations.

This script tests:
1. ChromaDB initialization
2. Collection creation
3. Adding documents
4. Searching documents
5. Retrieving by IDs
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.vector_config import VectorConfig
from app.services.vector_store import VectorStore


def test_vector_store():
    """Test vector store operations."""
    print("=" * 60)
    print("Testing VectorStore")
    print("=" * 60)

    try:
        # 1. Initialize VectorStore
        print("\n1. Initializing VectorStore...")
        config = VectorConfig()
        vector_store = VectorStore(config)
        print(f"VectorStore initialized successfully")
        print(f"   Persist directory: {config.get_persist_directory}")
        print(f"   Collection name: {config.get_collection_name}")

        # 2. Get collection info
        print("\n2. Getting collection info...")
        info = vector_store.get_collection_info()
        print(f"Collection info retrieved:")
        print(f"   Collection: {info['collection_name']}")
        print(f"   Total chunks: {info['total_chunks']}")

        # 3. Test adding documents (with dummy embeddings)
        print("\n3. Testing document addition...")
        doc_id = "test_doc_001"
        test_chunks = [
            "This is the first chunk of text about economic development.",
            "This is the second chunk discussing market trends.",
            "The third chunk contains information about business growth.",
        ]
        # Create dummy embeddings (384 dimensions)
        dummy_embeddings = [
            [0.1 * (i + 1) for _ in range(384)] for i in range(len(test_chunks))
        ]

        chunk_ids = vector_store.add_documents(
            doc_id=doc_id,
            chunks=test_chunks,
            embeddings=dummy_embeddings,
        )
        print(f"Added {len(chunk_ids)} chunks")
        print(f"   Chunk IDs: {chunk_ids[:3]}...")

        # 4. Test search
        print("\n4. Testing search functionality...")
        query_embedding = [0.1 for _ in range(384)]  # Dummy query embedding
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=3,
        )
        print(f"Search completed, found {len(results)} results")
        for i, result in enumerate(results[:2], 1):
            print(f"   Result {i}:")
            print(f"     ID: {result['id']}")
            print(f"     Text: {result['text'][:50]}...")
            print(f"     Similarity: {result['similarity_score']:.3f}")

        # 5. Test retrieval by IDs
        print("\n5. Testing retrieval by IDs...")
        retrieved = vector_store.get_by_ids(chunk_ids[:2])
        print(f"Retrieved {len(retrieved)} chunks by IDs")
        for i, chunk in enumerate(retrieved, 1):
            print(f"   Chunk {i}: {chunk['text'][:50]}...")

        # 6. Test collection info after addition
        print("\n6. Checking collection info after addition...")
        info_after = vector_store.get_collection_info()
        print(f"Collection now has {info_after['total_chunks']} chunks")

        # 7. Test document deletion
        print("\n7. Testing document deletion...")
        deleted = vector_store.delete_document(doc_id)
        print(f"Document deletion: {deleted}")

        # 8. Final collection info
        print("\n8. Final collection info...")
        final_info = vector_store.get_collection_info()
        print(f"Final chunk count: {final_info['total_chunks']}")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vector_store()
    sys.exit(0 if success else 1)

