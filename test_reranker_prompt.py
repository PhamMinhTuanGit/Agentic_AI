#!/usr/bin/env python3
"""
Test Updated Reranker with New RAG Prompt Format
=================================================

Tests the comprehensive reranking prompt that uses:
- Detailed grading scale (0-10)
- JSON object format: {"id0":8, "id1":6, ...}
- Only returns passages scoring 5+
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.reranker import LLMReranker
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_reranker_with_new_prompt():
    """Test the updated reranker with comprehensive RAG prompt"""
    
    print("\n" + "=" * 80)
    print("TESTING UPDATED RERANKER WITH RAG PROMPT FORMAT")
    print("=" * 80)
    
    # Initialize reranker
    print("\n[1] Initializing LLM Reranker...")
    reranker = LLMReranker(
        model="qwen2.5-coder:3b",
        top_k=5,
        temperature=0.1,
        timeout=120
    )
    print("✅ Reranker initialized")
    
    # Test query
    query = "How do I configure OSPF area 0 on a router?"
    
    # Sample documents (simulating retriever output)
    documents = [
        {
            'text': """OSPF (Open Shortest Path First) is a link-state routing protocol. 
            To configure OSPF area 0 (backbone area), use the 'router ospf' command followed 
            by the process ID. Then use 'network' statements to specify which networks 
            participate in OSPF. Example:
            configure
            router ospf 1
              network 10.0.0.0 0.255.255.255 area 0.0.0.0
              exit""",
            'score': 0.85
        },
        {
            'text': """BGP (Border Gateway Protocol) is used for inter-AS routing. 
            Configure BGP with 'router bgp' command and specify the AS number. 
            Use 'neighbor' commands to establish BGP sessions.""",
            'score': 0.45
        },
        {
            'text': """OSPF uses areas to reduce routing overhead. Area 0 is the backbone 
            area and all other areas must connect to it. The router-id is automatically 
            selected from loopback or highest IP address.""",
            'score': 0.75
        },
        {
            'text': """To view OSPF configuration, use 'show ip ospf' or 'show ip ospf interface'. 
            These commands display OSPF process information, neighbor states, and interface details.""",
            'score': 0.60
        },
        {
            'text': """Static routes are configured using the 'ip route' command. 
            Static routing is simple but doesn't adapt to network changes automatically.""",
            'score': 0.30
        }
    ]
    
    print(f"\n[2] Test Query: {query}")
    print(f"    Documents to rerank: {len(documents)}")
    
    # Show original ranking
    print("\n[3] Original Retriever Ranking:")
    for i, doc in enumerate(documents, 1):
        snippet = doc['text'][:80].replace('\n', ' ') + "..."
        print(f"    {i}. Score: {doc['score']:.2f} - {snippet}")
    
    # Rerank
    print("\n[4] Reranking with comprehensive RAG prompt...")
    print("    Expected: OSPF configuration doc ranks highest")
    print("    Expected: BGP and static route docs rank lowest or excluded")
    
    reranked_docs = reranker.rerank(query, documents, top_k=5)
    
    # Show reranked results
    print("\n[5] Reranked Results:")
    if reranked_docs:
        for doc in reranked_docs:
            snippet = doc['text'][:80].replace('\n', ' ') + "..."
            llm_score = doc.get('llm_score', 0)
            original_rank = doc.get('original_rank', 0)
            retriever_score = doc.get('score', 0)
            
            print(f"\n    Rank {doc.get('reranked_rank', '?')}:")
            print(f"      LLM Score: {llm_score:.1f}/100")
            print(f"      Original Rank: #{original_rank} (Retriever: {retriever_score:.2f})")
            print(f"      Text: {snippet}")
    else:
        print("    ❌ No documents returned!")
    
    # Validation
    print("\n[6] Validation:")
    if reranked_docs:
        top_doc = reranked_docs[0]
        top_text = top_doc['text'].lower()
        
        # Check if OSPF configuration doc is ranked highest
        if 'router ospf' in top_text and 'network' in top_text and 'area 0' in top_text:
            print("    ✅ CORRECT: OSPF configuration doc ranked #1")
        else:
            print("    ⚠️  WARNING: OSPF configuration doc not ranked #1")
        
        # Check scoring scale
        top_score = top_doc.get('llm_score', 0)
        if 70 <= top_score <= 100:
            print(f"    ✅ CORRECT: Top score in expected range ({top_score:.1f}/100)")
        else:
            print(f"    ⚠️  WARNING: Top score unusual ({top_score:.1f}/100)")
        
        # Check minimum score threshold (>= 50)
        min_score = min([d.get('llm_score', 0) for d in reranked_docs])
        if min_score >= 50:
            print(f"    ✅ CORRECT: All returned docs score >= 50 (min: {min_score:.1f})")
        else:
            print(f"    ❌ FAILED: Some docs scored < 50 (min: {min_score:.1f})")
        
        # Check if irrelevant docs were filtered out
        all_scores = [d.get('llm_score', 0) for d in reranked_docs]
        if all(score >= 50 for score in all_scores):
            print(f"    ✅ CORRECT: Quality threshold applied (all >= 50)")
        else:
            print(f"    ⚠️  WARNING: Some low-quality docs included")
    else:
        print("    ℹ️  INFO: No documents to validate (all scored < 50)")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Query: {query}")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Documents returned: {len(reranked_docs)}")
    if reranked_docs:
        scores = [d.get('llm_score', 0) for d in reranked_docs]
        print(f"   Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"   Average score: {sum(scores)/len(scores):.1f}")
    
    print("\n✅ Reranker using new RAG prompt format")
    print("   - Comprehensive grading scale (0-10 → 0-100)")
    print("   - JSON object format: {\"id0\":8, \"id1\":6, ...}")
    print("   - Only returns passages scoring 5+ (50+ on 0-100 scale)")
    print("   - Detailed evaluation criteria")
    print("   - Quality threshold: Filters out docs with score < 50/100")


if __name__ == "__main__":
    try:
        test_reranker_with_new_prompt()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
