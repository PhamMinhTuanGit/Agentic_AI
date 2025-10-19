"""
Test Retriever-Reranker Pipeline:
1. Retrieve top 10 documents using HybridRetriever
2. Rerank using LLM (qwen2.5-coder:3b) to get top 5
3. Display results
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add agent module to path
sys.path.insert(0, os.path.dirname(__file__))

from agent.retriever import HybridRetriever
from agent.reranker import LLMReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RetrieverRerankerPipeline:
    """
    Pipeline combining retriever and reranker for optimal document ranking
    """
    
    def __init__(self,
                 retriever_top_k: int = 10,
                 reranker_top_k: int = 5,
                 embedding_model: str = "nomic-embed-text",
                 rerank_model: str = "qwen2.5-coder:3b"):
        """
        Initialize pipeline
        
        Args:
            retriever_top_k: Number of documents to retrieve
            reranker_top_k: Number of final documents after reranking
            embedding_model: Dense embedding model
            rerank_model: LLM reranking model
        """
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
        
        logger.info("🚀 Initializing Retriever-Reranker Pipeline")
        logger.info("=" * 60)
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            faiss_index_path="database/document/hybrid_docs_index.faiss",
            metadata_path="database/document/hybrid_docs_metadata.json",
            tfidf_path="database/document/tfidf_vectorizer.pkl",
            svd_path="database/document/svd_transformer.pkl",
            embedding_model=embedding_model,
            top_k=retriever_top_k
        )
        
        # Initialize reranker
        self.reranker = LLMReranker(
            model=rerank_model,
            top_k=reranker_top_k,
            temperature=0.1,
            timeout=60
        )
        
        logger.info("✅ Pipeline initialized successfully!")
        logger.info("=" * 60)
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process query through full pipeline
        
        Args:
            query: Input query
        
        Returns:
            Dictionary with retriever and reranker results
        """
        logger.info(f"\n📝 Query: {query}")
        logger.info("=" * 60)
        
        # Step 1: Retrieval
        logger.info(f"\n[STEP 1] 🔍 RETRIEVAL (top-{self.retriever_top_k})")
        logger.info("-" * 60)
        
        retrieved_docs = self.retriever.retrieve_with_scores(query, top_k=self.retriever_top_k)
        
        if not retrieved_docs:
            logger.error("❌ No documents retrieved")
            return {
                'query': query,
                'retrieved_count': 0,
                'reranked_count': 0,
                'retrieved_docs': [],
                'final_docs': []
            }
        
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents\n")
        
        # Display retrieval results
        for doc in retrieved_docs:
            logger.info(f"  [{doc['rank']}] Retriever Score: {doc['score']:.4f}")
            logger.info(f"      Distance: {doc['distance']:.4f}")
            logger.info(f"      Text: {doc['text'][:80]}...")
        
        # Step 2: Reranking
        logger.info(f"\n[STEP 2] 🤖 RERANKING (using {self.reranker.model})")
        logger.info(f"         Selecting top-{self.reranker_top_k} from {len(retrieved_docs)}")
        logger.info("-" * 60)
        
        final_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.reranker_top_k)
        
        if not final_docs:
            logger.error("❌ Reranking failed")
            return {
                'query': query,
                'retrieved_count': len(retrieved_docs),
                'reranked_count': 0,
                'retrieved_docs': retrieved_docs,
                'final_docs': []
            }
        
        # Display reranking results
        logger.info(f"\n✅ Final Results (top-{self.reranker_top_k}):\n")
        
        for doc in final_docs:
            logger.info(f"  [Rank {doc['reranked_rank']}] 🏆 LLM Score: {doc['llm_score']:.1f}/100")
            logger.info(f"             Original Rank: {doc['original_rank']}")
            logger.info(f"             Retriever Score: {doc['score']:.4f}")
            logger.info(f"             Text: {doc['text'][:100]}...\n")
        
        return {
            'query': query,
            'retrieved_count': len(retrieved_docs),
            'reranked_count': len(final_docs),
            'retrieved_docs': retrieved_docs,
            'final_docs': final_docs
        }
    
    def process_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            queries: List of queries
        
        Returns:
            List of pipeline results
        """
        logger.info(f"\n🔄 Processing batch of {len(queries)} queries")
        logger.info("=" * 60)
        
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n[{i}/{len(queries)}] Processing query...")
            result = self.process(query)
            results.append(result)
        
        return results
    
    def print_summary(self, result: Dict[str, Any]):
        """Print summary of pipeline results"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Query: {result['query']}")
        logger.info(f"Documents Retrieved: {result['retrieved_count']}")
        logger.info(f"Documents After Reranking: {result['reranked_count']}")
        
        if result['final_docs']:
            logger.info(f"\n🏆 TOP RESULT:")
            top_doc = result['final_docs'][0]
            logger.info(f"   LLM Score: {top_doc['llm_score']:.1f}/100")
            logger.info(f"   Retriever Score: {top_doc['score']:.4f}")
            logger.info(f"   Text: {top_doc['text'][:150]}...\n")
    
    def print_stats(self):
        """Print pipeline statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("📈 PIPELINE CONFIGURATION")
        logger.info("=" * 60)
        
        logger.info("\n🔍 Retriever Config:")
        retriever_stats = self.retriever.get_stats()
        for key, value in retriever_stats.items():
            if key != 'metadata':
                logger.info(f"   {key}: {value}")
        
        logger.info("\n🤖 Reranker Config:")
        reranker_config = self.reranker.get_config()
        for key, value in reranker_config.items():
            logger.info(f"   {key}: {value}")


def main():
    """Main test function"""
    
    # Initialize pipeline
    pipeline = RetrieverRerankerPipeline(
        retriever_top_k=10,
        reranker_top_k=5,
        embedding_model="nomic-embed-text",
        rerank_model="qwen2.5-coder:3b"
    )
    
    # Print configuration
    pipeline.print_stats()
    
    # Test queries
    test_queries = [
        "What is Border Gateway Protocol and how does it work?",
        "How to configure BGP neighbors and establish connections?",
        "What are BGP commands for route manipulation?",
        "How to enable debugging in BGP?",
        "Explain BGP attributes and their usage"
    ]
    
    logger.info("\n\n" + "=" * 60)
    logger.info("🧪 RETRIEVER-RERANKER PIPELINE TEST")
    logger.info("=" * 60)
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n\n{'='*60}")
        logger.info(f"TEST CASE {i}/{len(test_queries)}")
        logger.info(f"{'='*60}")
        
        result = pipeline.process(query)
        pipeline.print_summary(result)
        
        # Optional: Save result to file
        # with open(f"pipeline_result_{i}.json", "w") as f:
        #     import json
        #     json.dump(result, f, indent=2, default=str)
    
    logger.info("\n\n" + "=" * 60)
    logger.info("✅ TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
