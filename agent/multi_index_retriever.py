#!/usr/bin/env python3
"""
Multi-Index Retriever for ZebOS RAG System
==========================================

Enhanced retriever that searches across multiple knowledge bases:
1. Main documentation (hybrid_docs)
2. ZebOS commands database (hybrid embeddings)
"""

import os
import json
import faiss
import pickle
import numpy as np
import requests
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from sklearn.preprocessing import normalize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")

from .retriever import HybridRetriever

logger = logging.getLogger(__name__)


class MultiIndexRetriever:
    """
    Multi-index retriever that searches across multiple knowledge bases
    and combines results intelligently
    """
    
    def __init__(
        self,
        main_index_path: str = "database/document/hybrid_docs_index.faiss",
        commands_index_dir: str = "database/commands",
        top_k: int = 5,
        commands_weight: float = 0.4
    ):
        """
        Initialize Multi-Index Retriever
        
        Args:
            main_index_path: Path to main hybrid documents index
            commands_index_dir: Directory containing ZebOS commands index
            top_k: Number of results per index
            commands_weight: Weight for commands results (0-1)
        """
        self.top_k = top_k
        self.commands_weight = commands_weight
        self.main_weight = 1.0 - commands_weight
        
        # Initialize main hybrid retriever
        logger.info("🔄 Initializing main hybrid retriever...")
        try:
            self.main_retriever = HybridRetriever(
                faiss_index_path=main_index_path,
                top_k=top_k
            )
            logger.info("✅ Main retriever initialized")
        except Exception as e:
            logger.warning(f"⚠️  Main retriever initialization failed: {e}")
            self.main_retriever = None
        
        # Initialize commands retriever
        logger.info("🔄 Initializing commands retriever...")
        self.commands_retriever = None
        self.commands_index = None
        self.commands_texts = []
        self.commands_metadata = []
        self.commands_tfidf = None
        self.commands_svd = None
        
        if os.path.exists(commands_index_dir):
            try:
                # Load commands FAISS index
                index_path = os.path.join(commands_index_dir, "zebos_commands_index.faiss")
                if os.path.exists(index_path):
                    self.commands_index = faiss.read_index(index_path)
                    logger.info(f"✅ Loaded commands FAISS index: {self.commands_index.ntotal} vectors")
                
                # Load commands metadata
                metadata_path = os.path.join(commands_index_dir, "zebos_commands_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        self.commands_texts = metadata.get('texts', [])
                        self.commands_metadata = metadata.get('metadata', [])
                        logger.info(f"✅ Loaded commands metadata: {len(self.commands_texts)} texts")
                
                # Load TF-IDF vectorizer
                tfidf_path = os.path.join(commands_index_dir, "tfidf_vectorizer.pkl")
                if os.path.exists(tfidf_path):
                    with open(tfidf_path, 'rb') as f:
                        self.commands_tfidf = pickle.load(f)
                    logger.info("✅ Loaded commands TF-IDF vectorizer")
                
                # Load SVD transformer
                svd_path = os.path.join(commands_index_dir, "svd_transformer.pkl")
                if os.path.exists(svd_path):
                    with open(svd_path, 'rb') as f:
                        self.commands_svd = pickle.load(f)
                    logger.info("✅ Loaded commands SVD transformer")
                
                logger.info("✅ Commands retriever initialized")
            except Exception as e:
                logger.warning(f"⚠️  Commands retriever initialization failed: {e}")
        else:
            logger.warning(f"⚠️  Commands index not found: {commands_index_dir}")
    
    def _search_commands(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search commands database using hybrid embeddings
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results from commands database
        """
        if not self.commands_index or not self.commands_texts:
            return []
        
        try:
            # Get dense embedding
            response = requests.post(
                EMBEDDING_API_URL,
                json={"model": "nomic-embed-text", "prompt": query},
                timeout=30
            )
            response.raise_for_status()
            dense_emb = np.array([response.json()["embedding"]], dtype='float32')
            
            # Get sparse embedding if vectorizer available
            if self.commands_tfidf and self.commands_svd:
                sparse_matrix = self.commands_tfidf.transform([query])
                sparse_emb = sparse_matrix.toarray().astype('float32')
                
                # Normalize
                dense_emb_norm = normalize(dense_emb, axis=1, norm='l2')
                sparse_emb_norm = normalize(sparse_emb, axis=1, norm='l2')
                
                # Apply SVD
                sparse_emb_reduced = self.commands_svd.transform(sparse_emb_norm)
                
                # Pad if needed
                if sparse_emb_reduced.shape[1] < dense_emb_norm.shape[1]:
                    padding = np.zeros((1, dense_emb_norm.shape[1] - sparse_emb_reduced.shape[1]))
                    sparse_emb_reduced = np.hstack([sparse_emb_reduced, padding])
                
                # Combine (alpha = 0.7 for dense)
                hybrid_emb = 0.7 * dense_emb_norm + 0.3 * sparse_emb_reduced
                query_vector = normalize(hybrid_emb, axis=1, norm='l2').astype('float32')
            else:
                # Use only dense embedding
                query_vector = normalize(dense_emb, axis=1, norm='l2').astype('float32')
            
            # Search FAISS index
            distances, indices = self.commands_index.search(query_vector, min(top_k, len(self.commands_texts)))
            
            # Format results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.commands_texts):
                    results.append({
                        'content': self.commands_texts[idx],
                        'metadata': self.commands_metadata[idx] if idx < len(self.commands_metadata) else {},
                        'score': float(dist),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching commands: {e}")
            return []
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_commands: bool = True,
        include_main: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search across all indices and combine results
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides default)
            include_commands: Whether to search commands database
            include_main: Whether to search main documentation
            
        Returns:
            Combined and ranked search results
        """
        if top_k is None:
            top_k = self.top_k
        
        all_results = []
        
        # Search main documentation
        if include_main and self.main_retriever:
            try:
                logger.info("🔍 Searching main documentation...")
                main_results = self.main_retriever.retrieve_with_scores(query, top_k=top_k)
                
                # Add source and weight
                for result in main_results:
                    result['source'] = 'main_docs'
                    result['weighted_score'] = result.get('score', 0.5) * self.main_weight
                    all_results.append(result)
                
                logger.info(f"   Found {len(main_results)} results from main docs")
            except Exception as e:
                logger.error(f"❌ Error searching main docs: {e}")
        
        # Search commands database
        if include_commands and self.commands_index:
            try:
                logger.info("🔍 Searching commands database...")
                commands_results = self._search_commands(query, top_k=top_k)
                
                # Add source and weight
                for result in commands_results:
                    result['source'] = 'commands_db'
                    result['weighted_score'] = result.get('score', 0.5) * self.commands_weight
                    all_results.append(result)
                
                logger.info(f"   Found {len(commands_results)} results from commands")
            except Exception as e:
                logger.error(f"❌ Error searching commands: {e}")
        
        # Sort by weighted score
        all_results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        # Return top-k combined results
        combined_results = all_results[:top_k]
        
        logger.info(f"✅ Combined {len(combined_results)} total results")
        
        return combined_results
    
    def search_commands_only(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search only the commands database
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results from commands database
        """
        return self.search(query, top_k=top_k, include_commands=True, include_main=False)
    
    def search_docs_only(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search only the main documentation
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results from main documentation
        """
        return self.search(query, top_k=top_k, include_commands=False, include_main=True)
    
    def format_context(self, results: List[Dict[str, Any]], max_length: int = 4000) -> str:
        """
        Format search results into context for LLM
        
        Args:
            results: Search results
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Get source indicator
            source = result.get('source', 'unknown')
            source_label = "📄 Main Docs" if source == 'main_docs' else "⚡ ZebOS Commands"
            
            # Get metadata
            metadata = result.get('metadata', {})
            meta_str = ""
            
            if metadata.get('type') == 'command':
                meta_str = f"[Command: {metadata.get('command_name', 'unknown')}]"
            elif metadata.get('type') == 'chapter':
                meta_str = f"[Chapter: {metadata.get('title', 'unknown')}]"
            
            # Format result
            content = result.get('content', '')
            score = result.get('score', 0)
            
            part = f"""
{source_label} - Result {i} {meta_str}
Score: {score:.3f}
---
{content}
---
"""
            
            # Check length
            if current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return '\n'.join(context_parts)


def test_multi_index_retriever():
    """Test the multi-index retriever"""
    print("=" * 70)
    print("TESTING MULTI-INDEX RETRIEVER")
    print("=" * 70)
    
    # Initialize retriever
    retriever = MultiIndexRetriever()
    
    # Test queries
    test_queries = [
        "How to configure OSPF on router?",
        "What is the syntax for BGP neighbor command?",
        "Configure VLAN on switch",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        results = retriever.search(query, top_k=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {result.get('source', 'unknown')}")
            print(f"Score: {result.get('score', 0):.3f}")
            print(f"Weighted Score: {result.get('weighted_score', 0):.3f}")
            
            metadata = result.get('metadata', {})
            if metadata:
                print(f"Metadata: {metadata}")
            
            content = result.get('content', '')
            print(f"Content (first 200 chars):\n{content[:200]}...")


if __name__ == "__main__":
    test_multi_index_retriever()
