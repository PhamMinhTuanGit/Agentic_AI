import os
import json
import pickle
import logging
import numpy as np
import requests
import faiss
from typing import List, Dict, Tuple, Any, Optional
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "../database/document/hybrid_docs_index.faiss")
METADATA_PATH = os.getenv("METADATA_PATH", "../database/document/hybrid_docs_metadata.json")
TFIDF_PATH = os.getenv("TFIDF_PATH", "../database/document/tfidf_vectorizer.pkl")
SVD_PATH = os.getenv("SVD_PATH", "../database/document/svd_transformer.pkl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retriever for Agentic system using hybrid embeddings (dense + sparse with SVD alignment)
    """
    
    def __init__(self, 
                 faiss_index_path: str = FAISS_INDEX_PATH,
                 metadata_path: str = METADATA_PATH,
                 tfidf_path: str = TFIDF_PATH,
                 svd_path: str = SVD_PATH,
                 embedding_model: str = "nomic-embed-text",
                 top_k: int = 5):
        """
        Initialize the Hybrid Retriever
        
        Args:
            faiss_index_path: Path to FAISS index
            metadata_path: Path to metadata JSON
            tfidf_path: Path to TF-IDF vectorizer
            svd_path: Path to SVD transformer
            embedding_model: Embedding model name
            top_k: Number of top results to return
        """
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.tfidf_path = tfidf_path
        self.svd_path = svd_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Storage
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[str, Any] = {}
        self.texts: List[str] = []
        self.tfidf_vectorizer = None
        self.svd_transformer = None
        
        # Load resources
        self._load_resources()
    
    def _load_resources(self) -> bool:
        """Load FAISS index, metadata, and transformers"""
        try:
            logger.info("🔄 Loading hybrid retriever resources...")
            
            # Load FAISS index
            if not os.path.exists(self.faiss_index_path):
                logger.error(f"❌ FAISS index not found: {self.faiss_index_path}")
                return False
            
            self.index = faiss.read_index(self.faiss_index_path)
            logger.info(f"✅ Loaded FAISS index from {self.faiss_index_path}")
            
            # Load metadata
            if not os.path.exists(self.metadata_path):
                logger.error(f"❌ Metadata not found: {self.metadata_path}")
                return False
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.metadata = metadata.get('config', {})
            self.texts = metadata.get('texts', [])
            logger.info(f"✅ Loaded metadata with {len(self.texts)} text chunks")
            logger.info(f"   Config: {self.metadata}")
            
            # Load TF-IDF vectorizer
            if os.path.exists(self.tfidf_path):
                with open(self.tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info(f"✅ Loaded TF-IDF vectorizer")
            else:
                logger.warning(f"⚠️ TF-IDF vectorizer not found: {self.tfidf_path}")
            
            # Load SVD transformer
            if os.path.exists(self.svd_path):
                with open(self.svd_path, 'rb') as f:
                    self.svd_transformer = pickle.load(f)
                logger.info(f"✅ Loaded SVD transformer")
            else:
                logger.warning(f"⚠️ SVD transformer not found: {self.svd_path}")
            
            logger.info("✅ All resources loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading resources: {e}")
            return False
    
    def _get_dense_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get dense embedding from Ollama API"""
        try:
            response = requests.post(
                EMBEDDING_API_URL,
                json={"model": self.embedding_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            
            if not embedding:
                logger.warning(f"⚠️ Empty embedding returned from API")
                return None
            
            return np.array(embedding, dtype='float32')
        except Exception as e:
            logger.error(f"❌ Error getting dense embedding: {e}")
            return None
    
    def _get_sparse_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get sparse (TF-IDF) embedding"""
        try:
            if self.tfidf_vectorizer is None:
                logger.warning("⚠️ TF-IDF vectorizer not loaded")
                return None
            
            sparse_matrix = self.tfidf_vectorizer.transform([text])
            sparse_embedding = sparse_matrix.toarray()[0].astype('float32')
            
            return sparse_embedding
        except Exception as e:
            logger.error(f"❌ Error getting sparse embedding: {e}")
            return None
    
    def _create_hybrid_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Create hybrid query embedding (dense + sparse with SVD alignment)
        """
        try:
            # Get dense embedding
            dense_emb = self._get_dense_embedding(text)
            if dense_emb is None:
                logger.error("❌ Failed to get dense embedding")
                return None
            
            # Get sparse embedding
            sparse_emb = self._get_sparse_embedding(text)
            if sparse_emb is None:
                logger.error("❌ Failed to get sparse embedding")
                return None
            
            # Align dimensions using SVD if needed
            if len(sparse_emb) != len(dense_emb):
                if self.svd_transformer is None:
                    logger.error("❌ SVD transformer not loaded, cannot align dimensions")
                    return None
                
                logger.debug(f"⚠️ Adjusting sparse embedding from {len(sparse_emb)} to {len(dense_emb)} dims")
                sparse_emb = self.svd_transformer.transform([sparse_emb])[0].astype('float32')
            
            # Normalize
            dense_norm = normalize([dense_emb], norm='l2')[0].astype('float32')
            sparse_norm = normalize([sparse_emb], norm='l2')[0].astype('float32')
            
            # Get alpha from metadata
            alpha = float(self.metadata.get('alpha', 0.7))
            
            # Combine: alpha * dense + (1-alpha) * sparse
            hybrid = (alpha * dense_norm + (1 - alpha) * sparse_norm).astype('float32')
            
            return hybrid
        except Exception as e:
            logger.error(f"❌ Error creating hybrid query embedding: {e}")
            return None
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query: Query text
            top_k: Number of results to return (uses self.top_k if not provided)
        
        Returns:
            List of dicts with 'text', 'distance', and 'rank' keys
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.index is None:
            logger.error("❌ FAISS index not loaded")
            return []
        
        try:
            logger.info(f"🔍 Retrieving {top_k} documents for query: {query[:50]}...")
            
            # Create hybrid query embedding
            query_embedding = self._create_hybrid_query_embedding(query)
            if query_embedding is None:
                logger.error("❌ Failed to create query embedding")
                return []
            
            # Search
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
                idx = int(idx)
                if idx < len(self.texts):
                    result = {
                        'rank': rank,
                        'text': self.texts[idx],
                        'distance': float(distance),
                        'index': idx
                    }
                    results.append(result)
                    logger.debug(f"  [{rank}] Distance: {distance:.4f}")
            
            logger.info(f"✅ Retrieved {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"❌ Error during retrieval: {e}")
            return []
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents with similarity scores (0-1)
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'text', 'score', 'distance', and 'rank' keys
        """
        results = self.retrieve(query, top_k)
        
        # Convert L2 distance to similarity score
        # Higher distance = lower similarity
        # Convert to range 0-1
        for result in results:
            # Using 1 / (1 + distance) formula to convert distance to similarity
            similarity = 1.0 / (1.0 + result['distance'])
            result['score'] = similarity
        
        return results
    
    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
        
        Returns:
            List of result lists
        """
        logger.info(f"🔄 Processing batch of {len(queries)} queries...")
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.debug(f"  [{i}/{len(queries)}] Processing query...")
            query_results = self.retrieve(query, top_k)
            results.append(query_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = {
            'total_documents': len(self.texts),
            'index_dimension': self.index.d if self.index else 0,
            'embedding_model': self.embedding_model,
            'default_top_k': self.top_k,
            'metadata': self.metadata,
            'has_tfidf_vectorizer': self.tfidf_vectorizer is not None,
            'has_svd_transformer': self.svd_transformer is not None
        }
        return stats
    
    def print_stats(self):
        """Print retriever statistics"""
        stats = self.get_stats()
        logger.info("📊 Retriever Statistics:")
        logger.info("=" * 50)
        for key, value in stats.items():
            if key != 'metadata':
                logger.info(f"  {key}: {value}")
        if stats['metadata']:
            logger.info("  Embedding Config:")
            for key, value in stats['metadata'].items():
                logger.info(f"    - {key}: {value}")


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_path="database/document/hybrid_docs_index.faiss",
        metadata_path="database/document/hybrid_docs_metadata.json",
        tfidf_path="database/document/tfidf_vectorizer.pkl",
        svd_path="database/document/svd_transformer.pkl",
        embedding_model="nomic-embed-text",
        top_k=5
    )
    
    # Print stats
    retriever.print_stats()
    
    # Example queries
    queries = [
        "How to configure BGP routes?",
        "What is Border Gateway Protocol?",
        "Command line interface help"
    ]
    
    logger.info("\n" + "=" * 50)
    logger.info("Testing Hybrid Retriever")
    logger.info("=" * 50 + "\n")
    
    for query in queries:
        logger.info(f"\n📝 Query: {query}")
        logger.info("-" * 50)
        
        results = retriever.retrieve_with_scores(query, top_k=3)
        
        for result in results:
            logger.info(f"\n[Rank {result['rank']}]")
            logger.info(f"  Score: {result['score']:.4f}")
            logger.info(f"  Distance: {result['distance']:.4f}")
            logger.info(f"  Text: {result['text'][:100]}...")
