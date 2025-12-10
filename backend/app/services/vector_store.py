"""
Vector Store Service

ChromaDB client wrapper for vector storage and retrieval operations.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from app.core.vector_config import VectorConfig
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB client wrapper for vector operations."""

    def __init__(self, config: Optional[VectorConfig] = None):
        """
        Initialize ChromaDB client.

        Args:
            config: Vector database configuration. If None, uses default config.
        """
        self.config = config or VectorConfig()
        
        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=self.config.get_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(
                f"ChromaDB client initialized at {self.config.get_persist_directory}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise

        # Get or create the default collection
        self.collection = self._get_or_create_collection(
            self.config.get_collection_name
        )

    def _get_or_create_collection(
        self, collection_name: str, embedding_dimension: int = 384
    ) -> chromadb.Collection:
        """
        Get existing collection or create a new one.

        Args:
            collection_name: Name of the collection
            embedding_dimension: Dimension of embeddings (default: 384)

        Returns:
            ChromaDB Collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
            return collection
        except Exception:
            # Collection doesn't exist, create it
            try:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Document chunks for RAG"},
                )
                logger.info(f"Created new collection: {collection_name}")
                return collection
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise

    def add_documents(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add document chunks to the vector store.

        Args:
            doc_id: Document identifier
            chunks: List of text chunks
            embeddings: List of embedding vectors (384 dimensions)
            metadatas: Optional list of metadata dictionaries for each chunk

        Returns:
            List of vector IDs for the added chunks
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return []

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        try:
            # Generate unique IDs for each chunk
            chunk_ids = [
                f"{doc_id}_chunk_{i}" for i in range(len(chunks))
            ]

            # Prepare metadata
            if metadatas is None:
                metadatas = [
                    {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "chunk_length": len(chunk),
                    }
                    for i, chunk in enumerate(chunks)
                ]
            else:
                # Ensure doc_id is in metadata
                for i, metadata in enumerate(metadatas):
                    metadata["doc_id"] = doc_id
                    metadata["chunk_index"] = i
                    if "chunk_length" not in metadata:
                        metadata["chunk_length"] = len(chunks[i])

            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            logger.info(
                f"Added {len(chunks)} chunks for document {doc_id} to vector store"
            )
            return chunk_ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar document chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            doc_id: Optional document ID to filter by
            where: Optional metadata filter dictionary

        Returns:
            List of search results with document text, metadata, and similarity scores
        """
        try:
            # Build where clause
            search_where = where or {}
            if doc_id:
                search_where["doc_id"] = doc_id

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=search_where if search_where else None,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score (ChromaDB returns distance)
                    # Distance is typically L2, so similarity = 1 / (1 + distance)
                    distance = results["distances"][0][i]
                    similarity = 1.0 / (1.0 + distance)

                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": similarity,
                        "distance": distance,
                    })

            logger.debug(
                f"Found {len(formatted_results)} results for query (top_k={top_k})"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise

    def get_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of chunk data with text and metadata
        """
        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )

            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    formatted_results.append({
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving chunks by IDs: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            True if deletion was successful
        """
        try:
            # Get all chunks for this document
            # Note: ids are always returned, no need to include them
            results = self.collection.get(
                where={"doc_id": doc_id},
            )

            if results["ids"]:
                # Delete all chunks
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {doc_id}"
                )
                return True
            else:
                logger.warning(f"No chunks found for document {doc_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection.name,
                "total_chunks": count,
                "persist_directory": self.config.get_persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def reset_collection(self) -> bool:
        """
        Reset (delete) the current collection.

        Returns:
            True if reset was successful
        """
        try:
            self.client.delete_collection(name=self.collection.name)
            # Recreate the collection
            self.collection = self._get_or_create_collection(
                self.config.get_collection_name
            )
            logger.info(f"Reset collection: {self.collection.name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

