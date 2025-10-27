#!/usr/bin/env python3
"""
Test Two-Stage RAG Pipeline
============================

Test the redesigned pipeline with two-stage retrieval:
1. Stage 1: Hybrid search in main documentation
2. Rerank documentation chunks
3. Detect commands mentioned
4. Stage 2: Search commands database for exact syntax
5. Build combined context
6. Generate answer with LLM

Test cases:
- OSPF configuration
- BGP configuration  
- Interface configuration
- Show commands
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.pipeline import RAGPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_two_stage_pipeline():
    """Test the two-stage retrieval pipeline"""
    
    print("\n" + "=" * 80)
    print("TWO-STAGE RAG PIPELINE TEST")
    print("=" * 80)
    
    # Initialize pipeline with multi-index enabled
    print("\n[1] Initializing pipeline...")
    print("   - Multi-index: ENABLED")
    print("   - Commands database: database/commands/")
    print("   - Two-stage retrieval: ACTIVE")
    
    pipeline = RAGPipeline(
        # Retriever config
        retriever_top_k=10,
        
        # Multi-index config (CRITICAL for two-stage)
        enable_multi_index=True,
        commands_index_dir="database/commands",
        commands_weight=0.4,
        
        # Reranker config
        reranker_top_k=5,
        
        # LLM config
        llm_model="qwen2.5-coder:3b",
        llm_temperature=0.1,
        
        # Cache config
        enable_cache=False,  # Disable cache for testing
        
        # Topology
        enable_topology=True,
        topology_file="network_stat/ring_topology.yaml",
        
        # CLI format
        enable_cli_format=True,
        cli_output_format="multi_code_block",
        
        # Chain-of-Thought
        enable_cot=False  # Disable CoT for cleaner test output
    )
    
    print("✅ Pipeline initialized!")
    
    # Test cases
    test_cases = [
        {
            "name": "OSPF Configuration",
            "question": "How do I configure OSPF area 0 on router R1?",
            "expected_commands": ["router ospf", "network", "area"]
        },
        {
            "name": "BGP Configuration",
            "question": "Configure BGP AS 65001 with neighbor 10.0.0.2",
            "expected_commands": ["router bgp", "neighbor", "bgp router-id"]
        },
        {
            "name": "Interface Configuration",
            "question": "Configure IP address 192.168.1.1/24 on ethernet 0/0",
            "expected_commands": ["interface ethernet", "ipv4 address", "no shutdown"]
        }
    ]
    
    # Run test cases
    for i, test in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST CASE {i}: {test['name']}")
        print("=" * 80)
        print(f"Question: {test['question']}")
        print(f"Expected commands: {', '.join(test['expected_commands'])}")
        print("-" * 80)
        
        # Query pipeline
        result = pipeline.query(
            question=test['question'],
            return_context=True,
            return_sources=True
        )
        
        # Display results
        print("\n📊 RESULTS:")
        print(f"   From cache: {result.get('from_cache', False)}")
        print(f"   Total time: {result.get('elapsed_time', 0):.2f}s")
        
        if 'breakdown' in result:
            breakdown = result['breakdown']
            print(f"\n   Time breakdown:")
            print(f"      Retrieval: {breakdown.get('retrieval', 0):.2f}s")
            print(f"      Reranking: {breakdown.get('reranking', 0):.2f}s")
            print(f"      Generation: {breakdown.get('generation', 0):.2f}s")
        
        print(f"\n   Model: {result.get('model', 'unknown')}")
        print(f"   Tokens: {result.get('tokens', 0)}")
        
        # Check if context has both sections
        if 'context' in result:
            context = result['context']
            has_docs = "DOCUMENTATION CONTEXT" in context
            has_commands = "COMMAND SYNTAX REFERENCE" in context
            has_detected = "DETECTED COMMANDS" in context
            
            print(f"\n   Context sections:")
            print(f"      ✅ Documentation context" if has_docs else "      ❌ Documentation context")
            print(f"      ✅ Detected commands" if has_detected else "      ❌ Detected commands")
            print(f"      ✅ Command syntax reference" if has_commands else "      ❌ Command syntax reference")
        
        # Display answer
        print("\n📝 ANSWER:")
        print("-" * 80)
        print(result['answer'])
        print("-" * 80)
        
        # Verify expected commands appear in answer
        answer_lower = result['answer'].lower()
        found_commands = []
        missing_commands = []
        
        for cmd in test['expected_commands']:
            if cmd.lower() in answer_lower:
                found_commands.append(cmd)
            else:
                missing_commands.append(cmd)
        
        print("\n✅ VALIDATION:")
        if found_commands:
            print(f"   Found commands: {', '.join(found_commands)}")
        if missing_commands:
            print(f"   Missing commands: {', '.join(missing_commands)}")
        
        # Verify ZebOS syntax (not Cisco)
        cisco_indicators = ["configure terminal", "ip address", "router-id", "network mask"]
        zebos_indicators = ["configure", "ipv4 address", "router ospf", "area"]
        
        cisco_found = [cmd for cmd in cisco_indicators if cmd in answer_lower]
        zebos_found = [cmd for cmd in zebos_indicators if cmd in answer_lower]
        
        if cisco_found:
            print(f"   ⚠️  WARNING: Cisco syntax detected: {', '.join(cisco_found)}")
        if zebos_found:
            print(f"   ✅ ZebOS syntax confirmed: {', '.join(zebos_found)}")
        
        print("\n" + "=" * 80)
        
        # Wait for user to continue
        if i < len(test_cases):
            input("\nPress Enter to continue to next test case...")
    
    # Display statistics
    print("\n" + "=" * 80)
    print("PIPELINE STATISTICS")
    print("=" * 80)
    stats = pipeline.get_stats()
    
    print(f"\nQueries processed: {stats.get('total_queries', 0)}")
    print(f"Cache hits: {stats.get('cache_hits', 0)}")
    print(f"Cache misses: {stats.get('cache_misses', 0)}")
    print(f"Total time: {stats.get('total_time', 0):.2f}s")
    print(f"Avg retrieval time: {stats.get('avg_retrieval_time', 0):.2f}s")
    print(f"Avg rerank time: {stats.get('avg_rerank_time', 0):.2f}s")
    print(f"Avg generation time: {stats.get('avg_generation_time', 0):.2f}s")
    
    print("\n" + "=" * 80)
    print("✅ TWO-STAGE PIPELINE TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_two_stage_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
