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
from agent.multi_index_retriever import MultiIndexRetriever
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
                 
                 # Multi-index retriever config
                 enable_multi_index: bool = True,
                 commands_index_dir: str = "database/commands",
                 commands_weight: float = 0.4,
                 
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
        
        # Initialize components - Multi-index or single retriever
        logger.info("\n[1/4] 🔍 Initializing Retriever...")
        self.enable_multi_index = enable_multi_index
        
        if enable_multi_index:
            # Use multi-index retriever (main docs + commands)
            try:
                self.retriever = MultiIndexRetriever(
                    main_index_path=faiss_index_path,
                    commands_index_dir=commands_index_dir,
                    top_k=retriever_top_k,
                    commands_weight=commands_weight
                )
                logger.info(f"✅ Multi-index retriever initialized (commands weight: {commands_weight})")
            except Exception as e:
                logger.warning(f"⚠️  Multi-index retriever failed, falling back to single index: {e}")
                self.retriever = HybridRetriever(
                    faiss_index_path=faiss_index_path,
                    metadata_path=metadata_path,
                    tfidf_path=tfidf_path,
                    svd_path=svd_path,
                    embedding_model=embedding_model,
                    top_k=retriever_top_k
                )
                self.enable_multi_index = False
        else:
            # Use single hybrid retriever (backward compatible)
            self.retriever = HybridRetriever(
                faiss_index_path=faiss_index_path,
                metadata_path=metadata_path,
                tfidf_path=tfidf_path,
                svd_path=svd_path,
                embedding_model=embedding_model,
                top_k=retriever_top_k
            )
            logger.info("ℹ️  Using single hybrid retriever")
        
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
    
    def _extract_command_mentions(self, documents: List[Dict[str, Any]], question: str) -> List[str]:
        """
        Extract ZebOS command mentions from reranked documents and question
        
        Detects command keywords like:
        - router ospf, router bgp, router rip
        - interface ethernet, interface loopback
        - ipv4 address, ipv6 address
        - neighbor, network, redistribute
        - show commands
        
        Args:
            documents: Reranked documentation chunks
            question: User's question
        
        Returns:
            List of detected command names/keywords
        """
        import re
        
        detected = set()
        
        # Common ZebOS command patterns
        command_patterns = [
            # Router protocols
            r'\b(router\s+(?:ospf|bgp|rip|isis)(?:\s+\d+)?)\b',
            # Interface commands
            r'\b(interface\s+(?:ethernet|loopback|gigabitethernet|tunnel)(?:\s+[\d/]+)?)\b',
            # IP address commands
            r'\b(ipv[46]\s+address)\b',
            # BGP commands
            r'\b(neighbor(?:\s+[\d.]+)?)\b',
            r'\b(bgp\s+(?:router-id|network|redistribute))\b',
            # OSPF commands
            r'\b(ospf\s+(?:area|network|passive-interface))\b',
            r'\b(area\s+[\d.]+)\b',
            # Show commands
            r'\b(show\s+(?:ip|ipv6|running-config|interface|route|bgp|ospf)(?:\s+\w+)*)\b',
            # General configuration
            r'\b(ip\s+(?:route|forwarding))\b',
            r'\b(no\s+shutdown)\b',
            r'\b(description)\b',
            r'\b(redistribute)\b',
            r'\b(network)\b',
        ]
        
        # Search in question
        question_lower = question.lower()
        for pattern in command_patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            detected.update(matches)
        
        # Search in document texts
        for doc in documents[:5]:  # Check top 5 reranked docs
            text_lower = doc.get('text', '').lower()
            for pattern in command_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                detected.update(matches)
        
        # Extract standalone command words from question
        command_keywords = [
            'router', 'interface', 'bgp', 'ospf', 'rip', 'isis',
            'neighbor', 'network', 'redistribute', 'area',
            'ipv4', 'ipv6', 'address', 'route', 'show',
            'configure', 'protocol', 'tunnel', 'loopback'
        ]
        
        words = question_lower.split()
        for keyword in command_keywords:
            if keyword in words:
                detected.add(keyword)
        
        return sorted(list(detected))
    
    def _build_two_stage_context(self, 
                                  doc_chunks: List[Dict[str, Any]], 
                                  command_chunks: List[Dict[str, Any]],
                                  detected_commands: List[str]) -> str:
        """
        Build combined context from two-stage retrieval
        
        Format:
        1. Network topology (if available)
        2. Detected commands summary
        3. Documentation context (conceptual info)
        4. Command syntax reference (exact syntax, parameters, examples)
        
        Args:
            doc_chunks: Reranked documentation chunks from Stage 1
            command_chunks: Command syntax chunks from Stage 2
            detected_commands: List of detected command keywords
        
        Returns:
            Formatted combined context string
        """
        context_parts = []
        
        # Section 1: Topology context
        if self.enable_topology and self.topology_context:
            context_parts.append("=" * 70)
            context_parts.append("NETWORK TOPOLOGY CONTEXT")
            context_parts.append("=" * 70)
            context_parts.append(self.topology_context)
            context_parts.append("=" * 70)
            context_parts.append("\n")
        
        # Section 2: Detected commands summary
        if detected_commands:
            context_parts.append("=" * 70)
            context_parts.append("DETECTED COMMANDS")
            context_parts.append("=" * 70)
            context_parts.append(f"The following commands were mentioned: {', '.join(detected_commands)}")
            context_parts.append("=" * 70)
            context_parts.append("\n")
        
        # Section 3: Documentation context
        context_parts.append("=" * 70)
        context_parts.append("DOCUMENTATION CONTEXT")
        context_parts.append("=" * 70)
        context_parts.append("The following documentation provides conceptual information:\n")
        
        for i, doc in enumerate(doc_chunks, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"\n[Document {i}] (Source: {source})")
            context_parts.append(doc['text'])
        
        context_parts.append("\n" + "=" * 70)
        context_parts.append("\n")
        
        # Section 4: Command syntax reference
        if command_chunks:
            context_parts.append("=" * 70)
            context_parts.append("COMMAND SYNTAX REFERENCE")
            context_parts.append("=" * 70)
            context_parts.append("The following are exact ZebOS command syntax, parameters, and examples:\n")
            
            for i, cmd_doc in enumerate(command_chunks, 1):
                # Commands have structured format
                text = cmd_doc.get('text', '')
                context_parts.append(f"\n[Command Reference {i}]")
                context_parts.append(text)
            
            context_parts.append("\n" + "=" * 70)
        
        return "\n".join(context_parts)
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from reranked documents (LEGACY - kept for backward compatibility)
        
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
        Process a query through the TWO-STAGE RAG pipeline
        
        Two-Stage Retrieval Process:
        1. Stage 1: Hybrid search in main documentation
           - Find relevant conceptual information
           - Detect what commands are needed
        2. Rerank the documentation chunks
        3. Stage 2: Search commands database
           - Get exact syntax, parameters, examples
           - Based on commands mentioned in Stage 1
        4. Combine contexts and generate answer
        
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
        logger.info(f"   Pipeline: Two-Stage Retrieval")
        logger.info("=" * 70)
        
        # Step 1: Check cache
        logger.info("\n[STEP 1/6] 💾 Checking cache...")
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
        logger.info("❌ Cache MISS - Processing through two-stage pipeline...")
        
        # STAGE 1: Retrieve from main documentation
        logger.info(f"\n[STEP 2/6] 🔍 STAGE 1: Searching main documentation...")
        logger.info(f"   Goal: Find relevant info & detect needed commands")
        retrieval_start = time.time()
        
        # Always search main docs first (not multi-index)
        if self.enable_multi_index and hasattr(self.retriever, 'main_retriever'):
            # Use main retriever from multi-index
            retrieved_docs = self.retriever.main_retriever.retrieve_with_scores(
                question, top_k=self.retriever_top_k
            )
        elif hasattr(self.retriever, 'retrieve_with_scores'):
            # Use single retriever
            retrieved_docs = self.retriever.retrieve_with_scores(
                question, top_k=self.retriever_top_k
            )
        else:
            logger.error("❌ No valid retriever available!")
            return {
                'question': question,
                'answer': "Error: Retrieval system not configured properly.",
                'from_cache': False,
                'error': 'No retriever available'
            }
        
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
        
        # STEP 3: Rerank documents
        logger.info(f"\n[STEP 3/6] 🤖 Reranking documentation to top-{self.reranker_top_k}...")
        rerank_start = time.time()
        
        reranked_docs = self.reranker.rerank(question, retrieved_docs, top_k=self.reranker_top_k)
        
        rerank_time = time.time() - rerank_start
        logger.info(f"✅ Reranked to {len(reranked_docs)} documents in {rerank_time:.2f}s")
        
        if not reranked_docs:
            logger.error("❌ Reranking failed!")
            reranked_docs = retrieved_docs[:self.reranker_top_k]
        
        # Extract command mentions from reranked docs
        logger.info(f"\n[STEP 4/6] 🔎 Detecting commands mentioned in documentation...")
        detected_commands = self._extract_command_mentions(reranked_docs, question)
        logger.info(f"✅ Detected {len(detected_commands)} command(s): {', '.join(detected_commands[:5])}")
        
        # STAGE 2: Search commands database for exact syntax
        commands_docs = []
        if self.enable_multi_index and detected_commands:
            logger.info(f"\n[STEP 4/6] ⚡ STAGE 2: Searching commands database...")
            logger.info(f"   Goal: Get exact syntax, parameters & examples")
            
            commands_search_start = time.time()
            
            # Search for each detected command
            for cmd in detected_commands[:10]:  # Limit to top 10 commands
                cmd_query = f"{cmd} command syntax parameters examples"
                if hasattr(self.retriever, '_search_commands'):
                    cmd_results = self.retriever._search_commands(cmd_query, top_k=2)
                    commands_docs.extend(cmd_results)
                    logger.info(f"   Found {len(cmd_results)} results for '{cmd}'")
            
            # Also do a general search with the original question
            if hasattr(self.retriever, '_search_commands'):
                general_cmd_results = self.retriever._search_commands(question, top_k=3)
                commands_docs.extend(general_cmd_results)
            
            commands_search_time = time.time() - commands_search_start
            logger.info(f"✅ Retrieved {len(commands_docs)} command documents in {commands_search_time:.2f}s")
        else:
            logger.info(f"\n[STEP 4/6] ℹ️  STAGE 2: Skipped (no multi-index or no commands detected)")
        
        # STEP 5: Build combined context
        logger.info(f"\n[STEP 5/6] 📝 Building combined context...")
        logger.info(f"   Documentation chunks: {len(reranked_docs)}")
        logger.info(f"   Command chunks: {len(commands_docs)}")
        
        context = self._build_two_stage_context(reranked_docs, commands_docs, detected_commands)
        logger.info(f"✅ Combined context built: {len(context)} characters")
        
        # STEP 6: Generate answer
        logger.info(f"\n[STEP 6/6] 💬 Generating answer with {self.llm_client.model}...")
        logger.info(f"   Format: {output_format}")
        generation_start = time.time()
        
        # Determine session type based on topology
        session_type = "topology" if self.enable_topology else "general"
        
        # Use CoT if enabled
        cot_prompt = None
        if self.enable_cot and self.cot:
            logger.info("   Using Chain-of-Thought reasoning...")
            analysis = self.cot.analyze_question(question)
            evaluated_docs = self.cot.evaluate_documents(question, reranked_docs)
            synthesis = self.cot.synthesize_information(question, evaluated_docs)
            plan = self.cot.plan_answer(question, synthesis)
            cot_prompt = self.cot.generate_cot_prompt(question, context, analysis, synthesis, plan)
            
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
            stats['cache_hit_rate'] = (stats['cache_hits'] / stats['total_queries']) * 100
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
