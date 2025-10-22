"""
RAG Pipeline with Caching
==========================

Pipeline trung tâm kết nối:
- Retriever: Tìm kiếm documents liên quan
- Reranker: Đánh giá lại độ liên quan bằng LLM
- LLM Client: Sinh câu trả lời
- Cache: Lưu trữ kết quả

Luồng xử lý:
1. Check cache → Cache HIT: Return cached answer
2. Cache MISS → Retrieve documents (top-K1)
3. Rerank documents (top-K2, K2 < K1)
4. Build context from top documents
5. Generate answer with LLM
6. Save to cache
7. Return result
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.retriever import HybridRetriever
from agent.reranker import LLMReranker
from rag.cache import CacheManager
from rag.llm_client import LLMClient
from rag.cli_output_config import CLIOutputConfig, create_cli_prompt
from rag.chain_of_thought import ChainOfThought
from network_stat.topology_parser import TopologyParser
from network_stat.network_rag import NetworkTopologyRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG Pipeline với caching
    
    Components:
    - Retriever: Hybrid retrieval (dense + sparse)
    - Reranker: LLM-based reranking
    - LLM: Answer generation
    - Cache: Query-answer caching
    """
    
    def __init__(self,
                 # Retriever config
                 retriever_top_k: int = 10,
                 embedding_model: str = "nomic-embed-text",
                 faiss_index_path: str = "database/document/hybrid_docs_index.faiss",
                 metadata_path: str = "database/document/hybrid_docs_metadata.json",
                 tfidf_path: str = "database/document/tfidf_vectorizer.pkl",
                 svd_path: str = "database/document/svd_transformer.pkl",
                 
                 # Reranker config
                 reranker_top_k: int = 5,
                 rerank_model: str = "qwen2.5-coder:3b",
                 
                 # LLM config
                 llm_model: str = "qwen2.5-coder:3b",
                 llm_temperature: float = 0.1,
                 llm_max_tokens: int = 4096,
                 
                 # Cache config
                 enable_cache: bool = True,
                 cache_dir: str = "cache",
                 cache_ttl_hours: int = 24,
                 
                 # Topology config
                 enable_topology: bool = True,
                 topology_file: str = "network_stat/ring_topology.yaml",
                 
                 # CLI Output config
                 enable_cli_format: bool = True,
                 cli_output_format: str = "multi_code_block",
                 
                 # Chain-of-Thought config
                 enable_cot: bool = True,
                 cot_debug: bool = True):
        """
        Initialize RAG Pipeline
        
        Args:
            retriever_top_k: Number of documents to retrieve
            embedding_model: Dense embedding model
            faiss_index_path: Path to FAISS index
            metadata_path: Path to metadata
            tfidf_path: Path to TF-IDF vectorizer
            svd_path: Path to SVD transformer
            reranker_top_k: Number of documents after reranking
            rerank_model: Reranking model
            llm_model: Answer generation model
            llm_temperature: LLM temperature
            llm_max_tokens: Max tokens in response
            enable_cache: Enable/disable caching
            cache_dir: Cache directory
            cache_ttl_hours: Cache TTL in hours
            enable_topology: Enable/disable topology context integration
            topology_file: Path to topology YAML file
            enable_cli_format: Enable/disable CLI output formatting
            cli_output_format: CLI output format (single_code_block)
        """
        logger.info("🚀 Initializing RAG Pipeline")
        logger.info("=" * 70)
        
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
        self.enable_topology = enable_topology
        self.enable_cli_format = enable_cli_format
        self.cli_output_format = cli_output_format
        self.cli_config = CLIOutputConfig() if enable_cli_format else None
        
        # Initialize topology if enabledoutput_format
        logger.info("\n[0/5] 🌐 Initializing Network Topology...")
        self.topology_parser = None
        self.network_rag = None
        self.topology_context = None
        
        if enable_topology:
            try:
                topology_path = Path(topology_file)
                if topology_path.exists():
                    self.topology_parser = TopologyParser(topology_file=str(topology_path))
                    logger.info(f"✅ Topology loaded from {topology_file}")
                    
                    # Build topology context for LLM
                    try:
                        self.network_rag = NetworkTopologyRAG(str(topology_path))
                        self.topology_context = self.network_rag.get_llm_context()
                        logger.info(f"✅ Topology context built ({len(self.topology_context)} characters)")
                    except Exception as e:
                        logger.warning(f"⚠️  Could not build full topology RAG context: {e}")
                        # Still try to get basic topology description
                        try:
                            self.topology_context = self.topology_parser.get_topology_description()
                            logger.info(f"✅ Using basic topology description instead")
                        except Exception as e2:
                            logger.warning(f"⚠️  Could not load topology description: {e2}")
                else:
                    logger.warning(f"⚠️  Topology file not found: {topology_file}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load topology: {e}")
        else:
            logger.info("ℹ️  Topology integration disabled")
        
        # Initialize components
        logger.info("\n[1/4] 🔍 Initializing Retriever...")
        self.retriever = HybridRetriever(
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
            tfidf_path=tfidf_path,
            svd_path=svd_path,
            embedding_model=embedding_model,
            top_k=retriever_top_k
        )
        
        logger.info("\n[2/4] 🤖 Initializing Reranker...")
        self.reranker = LLMReranker(
            model=rerank_model,
            top_k=reranker_top_k,
            temperature=0.1
        )
        
        logger.info("\n[3/4] 💬 Initializing LLM Client...")
        self.llm_client = LLMClient(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        
        logger.info("\n[4/4] 💾 Initializing Cache...")
        self.cache = CacheManager(
            cache_dir=cache_dir,
            ttl_hours=cache_ttl_hours,
            enable_cache=enable_cache
        )
        
        logger.info("\n[5/5] 🧠 Initializing Chain-of-Thought...")
        self.enable_cot = enable_cot
        self.cot = ChainOfThought(debug=cot_debug) if enable_cot else None
        
        if self.enable_cot:
            logger.info(f"✅ Chain-of-Thought enabled (debug={cot_debug})")
        else:
            logger.info("ℹ️  Chain-of-Thought disabled")
        
        # Pipeline statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_retrieval_time': 0.0,
            'avg_rerank_time': 0.0,
            'avg_generation_time': 0.0
        }
        
        logger.info("\n✅ RAG Pipeline initialized successfully!")
        logger.info("=" * 70)
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from reranked documents
        
        Includes:
        - Retrieved documents
        - Network topology information (if available)
        
        Args:
            documents: List of reranked documents
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add topology context if available
        if self.enable_topology and self.topology_context:
            context_parts.append("=" * 70)
            context_parts.append("NETWORK TOPOLOGY CONTEXT")
            context_parts.append("=" * 70)
            context_parts.append(self.topology_context)
            context_parts.append("=" * 70)
            context_parts.append("\n")
        
        # Add retrieved documents
        context_parts.append("=" * 70)
        context_parts.append("RELEVANT DOCUMENTS")
        context_parts.append("=" * 70)
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"\n[Document {i}]\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def query(self, 
             question: str,
             return_context: bool = False,
             return_sources: bool = False,
             output_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Luồng xử lý:
        1. Check cache
        2. Retrieve documents (if cache miss)
        3. Rerank documents
        4. Generate answer
        5. Save to cache
        
        Args:
            question: User question
            return_context: Include context in response
            return_sources: Include source documents in response
            output_format: Output format (default, single_code_block)
        
        Returns:
            Dict with answer and metadata
        """
        # Use configured output format if not specified
        if output_format is None:
            output_format = self.cli_output_format if self.enable_cli_format else "default"
        
        self.stats['total_queries'] += 1
        pipeline_start_time = time.time()
        
        logger.info("\n" + "=" * 70)
        logger.info(f"📝 QUERY: {question}")
        logger.info(f"   Output format: {output_format}")
        logger.info("=" * 70)
        
        # Step 1: Check cache
        logger.info("\n[STEP 1/5] 💾 Checking cache...")
        cached_result = self.cache.get(question)
        
        if cached_result:
            self.stats['cache_hits'] += 1
            elapsed_time = time.time() - pipeline_start_time
            
            logger.info(f"✅ Cache HIT! Returning cached answer")
            logger.info(f"⏱️  Total time: {elapsed_time:.2f}s")
            
            result = {
                'question': question,
                'answer': cached_result['answer'],
                'from_cache': True,
                'elapsed_time': elapsed_time
            }
            
            if return_context:
                result['context'] = cached_result.get('context', '')
            
            if return_sources:
                result['sources'] = cached_result.get('metadata', {}).get('sources', [])
            
            return result
        
        # Cache MISS - Continue with pipeline
        self.stats['cache_misses'] += 1
        logger.info("❌ Cache MISS - Processing through pipeline...")
        
        # Step 2: Retrieve documents
        logger.info(f"\n[STEP 2/5] 🔍 Retrieving top-{self.retriever_top_k} documents...")
        retrieval_start = time.time()
        
        retrieved_docs = self.retriever.retrieve_with_scores(question, top_k=self.retriever_top_k)
        
        retrieval_time = time.time() - retrieval_start
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
        
        if not retrieved_docs:
            logger.error("❌ No documents retrieved!")
            return {
                'question': question,
                'answer': "I couldn't find any relevant information to answer your question.",
                'from_cache': False,
                'error': 'No documents retrieved'
            }
        
        # Step 3: Rerank documents
        logger.info(f"\n[STEP 3/5] 🤖 Reranking to top-{self.reranker_top_k} documents...")
        rerank_start = time.time()
        
        reranked_docs = self.reranker.rerank(question, retrieved_docs, top_k=self.reranker_top_k)
        
        rerank_time = time.time() - rerank_start
        logger.info(f"✅ Reranked to {len(reranked_docs)} documents in {rerank_time:.2f}s")
        
        if not reranked_docs:
            logger.error("❌ Reranking failed!")
            reranked_docs = retrieved_docs[:self.reranker_top_k]
        
        # Step 4: Chain-of-Thought Reasoning (if enabled)
        cot_prompt = None
        if self.enable_cot and self.cot:
            logger.info(f"\n[STEP 4/5] 🧠 Chain-of-Thought Reasoning...")
            cot_start = time.time()
            
            # Run CoT analysis steps
            analysis = self.cot.analyze_question(question)
            evaluated_docs = self.cot.evaluate_documents(question, reranked_docs)
            synthesis = self.cot.synthesize_information(question, evaluated_docs)
            plan = self.cot.plan_answer(question, synthesis)
            
            # Build context
            context = self._build_context(reranked_docs)
            
            # Generate CoT-enhanced prompt
            cot_prompt = self.cot.generate_cot_prompt(question, context, analysis, synthesis, plan)
            
            cot_time = time.time() - cot_start
            logger.info(f"✅ Chain-of-Thought generated in {cot_time:.2f}s")
            logger.info(f"   Reasoning trace length: {len(self.cot.get_thoughts_summary())} characters")
            
            # Print thoughts for debugging
            logger.info("\n📋 REASONING THOUGHTS:")
            print("\n" + "=" * 70)
            print("📋 CHAIN-OF-THOUGHT REASONING TRACE")
            print("=" * 70)
            print(self.cot.get_thoughts_summary())
            print("=" * 70 + "\n")
        else:
            # Step 4: Build context (without CoT)
            logger.info(f"\n[STEP 4/5] 📝 Building context...")
            context = self._build_context(reranked_docs)
            logger.info(f"✅ Context built: {len(context)} characters")
        
        # Step 5: Generate answer
        logger.info(f"\n[STEP 5/5] 💬 Generating answer with {self.llm_client.model}...")
        logger.info(f"   Format: {output_format}")
        generation_start = time.time()
        
        # Determine session type based on topology
        session_type = "topology" if self.enable_topology else "general"
        
        # Use CoT prompt if available
        if cot_prompt:
            llm_result = self.llm_client.generate(
                query=question,
                context=context,
                output_format=output_format,
                session_type=session_type,
                use_cot=True,
                cot_prompt=cot_prompt
            )
        else:
            llm_result = self.llm_client.generate(
                query=question,
                context=context,
                output_format=output_format,
                session_type=session_type
            )
        
        generation_time = time.time() - generation_start
        logger.info(f"✅ Answer generated in {generation_time:.2f}s")
        
        # Calculate total time
        total_time = time.time() - pipeline_start_time
        
        # Update statistics
        self.stats['total_time'] += total_time
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['cache_misses'] - 1) + retrieval_time) 
            / self.stats['cache_misses']
        )
        self.stats['avg_rerank_time'] = (
            (self.stats['avg_rerank_time'] * (self.stats['cache_misses'] - 1) + rerank_time) 
            / self.stats['cache_misses']
        )
        self.stats['avg_generation_time'] = (
            (self.stats['avg_generation_time'] * (self.stats['cache_misses'] - 1) + generation_time) 
            / self.stats['cache_misses']
        )
        
        # Save to cache
        logger.info("\n💾 Saving to cache...")
        self.cache.set(
            query=question,
            answer=llm_result['answer'],
            context=context,
            metadata={
                'model': self.llm_client.model,
                'tokens': llm_result.get('total_tokens', 0),
                'sources': [{'text': doc['text'][:100], 'score': doc.get('llm_score', 0)} 
                           for doc in reranked_docs],
                'retrieval_time': retrieval_time,
                'rerank_time': rerank_time,
                'generation_time': generation_time
            }
        )
        
        # Build result
        result = {
            'question': question,
            'answer': llm_result['answer'],
            'from_cache': False,
            'elapsed_time': total_time,
            'breakdown': {
                'retrieval': retrieval_time,
                'reranking': rerank_time,
                'generation': generation_time
            },
            'model': self.llm_client.model,
            'tokens': llm_result.get('total_tokens', 0)
        }
        
        if return_context:
            result['context'] = context
        
        if return_sources:
            result['sources'] = [
                {
                    #'text': doc['text'][:200] + '...',
                    'text': doc['text'] + '...',
                    'llm_score': doc.get('llm_score', 0),
                    'retriever_score': doc.get('score', 0),
                    'rank': doc.get('reranked_rank', 0)
                }
                for doc in reranked_docs
            ]
        
        logger.info(f"\n⏱️  Total pipeline time: {total_time:.2f}s")
        logger.info(f"   ├─ Retrieval: {retrieval_time:.2f}s ({(retrieval_time/total_time)*100:.1f}%)")
        logger.info(f"   ├─ Reranking: {rerank_time:.2f}s ({(rerank_time/total_time)*100:.1f}%)")
        logger.info(f"   └─ Generation: {generation_time:.2f}s ({(generation_time/total_time)*100:.1f}%)")
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            questions: List of questions
        
        Returns:
            List of results
        """
        logger.info(f"\n🔄 Processing batch of {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"\n[{i}/{len(questions)}]")
            result = self.query(question)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.stats.copy()
        
        # Add cache stats
        cache_stats = self.cache.get_stats()
        stats['cache'] = cache_stats
        
        # Add LLM stats
        llm_stats = self.llm_client.get_stats()
        stats['llm'] = llm_stats
        
        # Calculate averages
        if stats['total_queries'] > 0:
            stats['avg_total_time'] = stats['total_time'] / stats['total_queries']
            stats['cache_hit_rate'] = (stats['cach:e_hits'] / stats['total_queries']) * 100
        else:
            stats['avg_total_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print pipeline statistics"""
        stats = self.get_stats()
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 RAG PIPELINE STATISTICS")
        logger.info("=" * 70)
        
        logger.info("\n📝 Query Statistics:")
        logger.info(f"  Total Queries: {stats['total_queries']}")
        logger.info(f"  Cache Hits: {stats['cache_hits']}")
        logger.info(f"  Cache Misses: {stats['cache_misses']}")
        logger.info(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2f}%")
        
        logger.info("\n⏱️  Performance:")
        logger.info(f"  Total Time: {stats['total_time']:.2f}s")
        logger.info(f"  Avg Time/Query: {stats['avg_total_time']:.2f}s")
        logger.info(f"  Avg Retrieval Time: {stats['avg_retrieval_time']:.2f}s")
        logger.info(f"  Avg Reranking Time: {stats['avg_rerank_time']:.2f}s")
        logger.info(f"  Avg Generation Time: {stats['avg_generation_time']:.2f}s")
        
        logger.info("\n💾 Cache Statistics:")
        logger.info(f"  Cache Size: {stats['cache']['cache_size']} entries")
        logger.info(f"  Total Accesses: {stats['cache']['total_accesses']}")
        
        logger.info("\n🤖 LLM Statistics:")
        logger.info(f"  Total Requests: {stats['llm']['total_requests']}")
        logger.info(f"  Success Rate: {stats['llm']['success_rate']:.2f}%")
        logger.info(f"  Total Tokens: {stats['llm']['total_tokens']}")
        logger.info(f"  Avg Tokens/Request: {stats['llm'].get('avg_tokens_per_request', 0):.0f}")
        
        logger.info("=" * 70 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline(
        retriever_top_k=10,
        reranker_top_k=5,
        enable_cache=True,
        cache_ttl_hours=24
    )
    
    # Test query
    question = "What is Border Gateway Protocol?"
    
    # First query - Cache MISS
    logger.info("\n=== TEST 1: First Query ===")
    result1 = pipeline.query(question, return_sources=True)
    print(f"\nAnswer: {result1['answer'][:200]}...")
    print(f"From cache: {result1['from_cache']}")
    print(f"Time: {result1['elapsed_time']:.2f}s")
    
    # Second query - Cache HIT
    logger.info("\n=== TEST 2: Same Query (Should hit cache) ===")
    result2 = pipeline.query(question)
    print(f"\nAnswer: {result2['answer'][:200]}...")
    print(f"From cache: {result2['from_cache']}")
    print(f"Time: {result2['elapsed_time']:.2f}s")
    
    # Print stats
    pipeline.print_stats()
