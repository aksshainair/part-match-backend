from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

class QdrantService:
    def __init__(self):
        """Initialize Qdrant client and collection name."""
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "parts_catalog"
        self.inv_po_collection_name = "inv_po_data_2"
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists, create if it doesn't."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=3072,  # For text-embedding-3-large
                        distance=models.Distance.COSINE
                    )
                }
            )

        if self.inv_po_collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.inv_po_collection_name,
                vectors_config={
                    "text": models.VectorParams(
                        size=3072,  # For text-embedding-3-large
                        distance=models.Distance.COSINE
                    )
                }
            )
    
    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any], vector: List[float]):
        """
        Add a document to the Qdrant collection.
        
        Args:
            document_id: Unique identifier for the document
            text: Document text content
            metadata: Additional metadata as a dictionary
            vector: Precomputed embedding vector
        """
        points = [
            models.PointStruct(
                id=document_id,
                vector={"text": vector},
                payload={
                    "text": text,
                    **metadata
                }
            )
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: List[float], top_k: int = 5, score_threshold: float = 0.6, collection_name: str = "parts_catalog"):
        """
        Search for similar documents in the Qdrant collection.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with scores
        """
        if collection_name == "parts_catalog":
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
        else:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=("text", query_vector),
                limit=top_k,
                score_threshold=score_threshold
        )
        
        return search_result
    
    def find_best_match(self, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Find the single best matching document.
        
        Args:
            query_vector: Query embedding vector
            
        Returns:
            Best matching document or None if no match found
        """
        results = self.search(query_vector, top_k=1)
        if not results:
            return None
            
        best_match = results[0]
        return {
            "id": best_match.id,
            "score": best_match.score,
            "payload": best_match.payload
        }
    
    def find_best_match_in_inv_po(self, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        """
        Find the single best matching document.
        
        Args:
            query_vector: Query embedding vector
            
        Returns:
            Best matching document or None if no match found
        """
        results = self.search(query_vector, top_k=1, collection_name=self.inv_po_collection_name)
        if not results:
            return None
            
        best_match = results[0]
        return {
            "id": best_match.id,
            "score": best_match.score,
            "payload": best_match.payload
        }
    
    def batch_search(self, query_vectors: List[List[float]], top_k: int = 5, score_threshold: float = 0.6):
        """
        Perform batch search with multiple query vectors.
        
        Args:
            query_vectors: List of query embedding vectors
            top_k: Number of results to return per query
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results for each query
        """
        search_queries = [
            models.SearchRequest(
                vector=("text", vector),
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold
            ) for vector in query_vectors
        ]
        
        return self.client.search_batch(
            collection_name=self.collection_name,
            requests=search_queries
        )

    def upsert_line_item_to_qdrant(self, doc_id: str, line_idx: int, embedding, metadata: dict):
        # Qdrant requires point IDs to be unsigned integers or UUIDs (not arbitrary strings)
        # We'll use a UUID for each line item
        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector={"text": embedding},  # Match the vector name 'text' from the collection config
            payload=metadata
        )
        self.client.upsert(
            collection_name=self.inv_po_collection_name,
            points=[point]
        )
