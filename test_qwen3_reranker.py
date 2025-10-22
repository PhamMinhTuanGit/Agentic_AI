#!/usr/bin/env python3
"""
Test Qwen3-Reranker-4B with updated reranker code
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.reranker import LLMReranker

def test_reranker():
    """Test reranker with specialized model"""
    print("\n" + "=" * 70)
    print("QWEN3 RERANKER TEST")
    print("=" * 70)
    
    # Initialize reranker with specialized model
    print("\n✅ Initializing reranker with Qwen3-Reranker-4B...")
    reranker = LLMReranker(
        model="dengcao/Qwen3-Reranker-4B:Q4_K_M",
        top_k=3,
        temperature=0.1
    )
    
    print(f"   Model: {reranker.model}")
    print(f"   API URL: {reranker.api_url}")
    print(f"   Top K: {reranker.top_k}")
    
    # Create test documents
    test_query = "What is OSPF routing protocol?"
    test_docs = [
        {
            'text': 'OSPF (Open Shortest Path First) is an interior gateway routing protocol used in IP networks. It uses the Dijkstra algorithm to calculate the shortest path.',
            'score': 0.85,
            'id': 1
        },
        {
            'text': 'BGP (Border Gateway Protocol) is used for routing between autonomous systems on the internet. It is an exterior gateway protocol.',
            'score': 0.70,
            'id': 2
        },
        {
            'text': 'RIP (Routing Information Protocol) is an older distance-vector routing protocol. It is rarely used in modern networks.',
            'score': 0.50,
            'id': 3
        },
        {
            'text': 'EIGRP (Enhanced Interior Gateway Routing Protocol) is a Cisco proprietary routing protocol.',
            'score': 0.60,
            'id': 4
        },
        {
            'text': 'Python is a programming language used for web development and data science.',
            'score': 0.10,
            'id': 5
        },
    ]
    
    print(f"\n📝 Test Query: \"{test_query}\"")
    print(f"📄 Test Documents: {len(test_docs)}")
    
    # Run reranker
    print(f"\n🔄 Running reranker...")
    try:
        reranked = reranker.rerank(test_query, test_docs, top_k=3)
        
        print(f"\n✅ Reranking successful!")
        print(f"   Results: {len(reranked)} documents")
        
        # Show results
        print(f"\n📊 Results:")
        for i, doc in enumerate(reranked, 1):
            llm_score = doc.get('llm_score', 'N/A')
            orig_score = doc.get('score', 'N/A')
            rank = doc.get('reranked_rank', 'N/A')
            text = doc['text'][:60] + '...'
            
            print(f"\n   [{i}] Rank: {rank}")
            print(f"       LLM Score: {llm_score}")
            print(f"       Original Score: {orig_score}")
            print(f"       Text: {text}")
        
        print("\n" + "=" * 70)
        print("✅ RERANKER TEST PASSED - Qwen3-Reranker-4B IS WORKING!")
        print("=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Reranking failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reranker()
    sys.exit(0 if success else 1)
