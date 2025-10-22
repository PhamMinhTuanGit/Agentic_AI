#!/usr/bin/env python3
"""
Test Chain-of-Thought Pipeline
==============================

Demonstrates CoT reasoning with debug output
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.pipeline import RAGPipeline


def test_cot_pipeline():
    """Test RAG pipeline with Chain-of-Thought reasoning enabled"""
    
    print("\n" + "=" * 70)
    print("🧠 CHAIN-OF-THOUGHT PIPELINE TEST")
    print("=" * 70 + "\n")
    
    # Initialize pipeline WITH Chain-of-Thought enabled
    print("📦 Initializing RAG Pipeline with CoT enabled...")
    pipeline = RAGPipeline(
        enable_topology=True,
        enable_cli_format=True,
        enable_cot=True,
        cot_debug=True  # This enables debug printing of thoughts
    )
    
    # Test queries
    test_queries = [
        "How do I configure OSPF routing protocol on ZebOS?",
        "What is the difference between BGP and OSPF?",
        "How do I setup interface configuration on a network switch?"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print("\n" + "=" * 70)
        print(f"QUERY {i}/{len(test_queries)}")
        print("=" * 70)
        print(f"❓ {question}\n")
        
        # Process query
        result = pipeline.query(
            question=question,
            return_context=False,
            return_sources=False,
            output_format="multi_code_block"
        )
        
        # Display result
        print("\n" + "=" * 70)
        print("✅ FINAL ANSWER")
        print("=" * 70)
        print(result['answer'])
        print("=" * 70)
        
        # Display timing breakdown
        if 'breakdown' in result:
            print("\n⏱️  Timing Breakdown:")
            for component, time_val in result['breakdown'].items():
                print(f"   - {component.capitalize()}: {time_val:.2f}s")
        
        print(f"\n📊 Total Time: {result['elapsed_time']:.2f}s")
        print(f"🔤 Tokens Used: {result.get('tokens', 0)}")
    
    # Display statistics
    print("\n" + "=" * 70)
    print("📈 PIPELINE STATISTICS")
    print("=" * 70)
    stats = pipeline.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"Cache Misses: {stats['cache_misses']}")
    print(f"Total Time: {stats['total_time']:.2f}s")
    
    if stats['total_queries'] > 0:
        print(f"Avg Retrieval Time: {stats['avg_retrieval_time']:.2f}s")
        print(f"Avg Rerank Time: {stats['avg_rerank_time']:.2f}s")
        print(f"Avg Generation Time: {stats['avg_generation_time']:.2f}s")


def test_cot_vs_no_cot():
    """Compare pipeline WITH and WITHOUT Chain-of-Thought"""
    
    print("\n" + "=" * 70)
    print("⚖️  COMPARING: CoT vs Non-CoT Pipeline")
    print("=" * 70 + "\n")
    
    test_question = "How do I configure OSPF routing protocol on ZebOS?"
    
    # Test WITHOUT CoT
    print("🔴 Testing WITHOUT Chain-of-Thought...")
    print("-" * 70)
    pipeline_no_cot = RAGPipeline(
        enable_topology=True,
        enable_cli_format=True,
        enable_cot=False  # Disabled
    )
    
    result_no_cot = pipeline_no_cot.query(
        question=test_question,
        return_context=False,
        return_sources=False,
        output_format="multi_code_block"
    )
    
    print(f"Time taken: {result_no_cot['elapsed_time']:.2f}s\n")
    
    # Test WITH CoT
    print("\n🟢 Testing WITH Chain-of-Thought...")
    print("-" * 70)
    pipeline_with_cot = RAGPipeline(
        enable_topology=True,
        enable_cli_format=True,
        enable_cot=True,  # Enabled
        cot_debug=False  # Silent debug (no printing)
    )
    
    result_with_cot = pipeline_with_cot.query(
        question=test_question,
        return_context=False,
        return_sources=False,
        output_format="multi_code_block"
    )
    
    print(f"Time taken: {result_with_cot['elapsed_time']:.2f}s\n")
    
    # Comparison
    print("\n" + "=" * 70)
    print("📊 COMPARISON RESULTS")
    print("=" * 70)
    print(f"Without CoT: {result_no_cot['elapsed_time']:.2f}s")
    print(f"With CoT:    {result_with_cot['elapsed_time']:.2f}s")
    
    time_diff = result_with_cot['elapsed_time'] - result_no_cot['elapsed_time']
    percent_diff = (time_diff / result_no_cot['elapsed_time']) * 100
    
    if time_diff > 0:
        print(f"Difference:  +{time_diff:.2f}s ({percent_diff:+.1f}%)")
        print("\n💡 Note: CoT reasoning adds some overhead but improves reasoning quality")
    else:
        print(f"Difference:  {time_diff:.2f}s ({percent_diff:+.1f}%)")
        print("\n💡 Note: CoT may cache results, making it faster")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Chain-of-Thought Pipeline")
    parser.add_argument('--compare', action='store_true', help='Compare CoT vs Non-CoT')
    args = parser.parse_args()
    
    if args.compare:
        test_cot_vs_no_cot()
    else:
        test_cot_pipeline()
