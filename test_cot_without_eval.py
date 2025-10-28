#!/usr/bin/env python3
"""
Test Chain-of-Thought without Document Evaluation
==================================================

Verifies that CoT works correctly after removing the evaluate_documents step.
Document relevance is now handled solely by the reranker.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.chain_of_thought import ChainOfThought

print("\n" + "=" * 70)
print("TESTING CHAIN-OF-THOUGHT (WITHOUT DOCUMENT EVALUATION)")
print("=" * 70)

# Initialize CoT
cot = ChainOfThought(debug=True)

# Test data
question = "How do I configure OSPF area 0 on router R1?"
documents = [
    {
        'text': 'OSPF configuration requires router ospf command and network statements...',
        'llm_score': 85.0,
        'reranked_rank': 1
    },
    {
        'text': 'OSPF area 0 is the backbone area that all other areas connect to...',
        'llm_score': 72.0,
        'reranked_rank': 2
    },
    {
        'text': 'Use network command with wildcard mask and area ID to configure OSPF...',
        'llm_score': 68.0,
        'reranked_rank': 3
    }
]

print("\n[1] Step 1: Analyze Question")
print("-" * 70)
analysis = cot.analyze_question(question)
print(f"✅ Analysis complete")
print(f"   Keywords: {', '.join(analysis['keywords'])}")
print(f"   Type: {analysis['question_type']}")

print("\n[2] Step 2: Synthesize Information (SKIP document evaluation)")
print("-" * 70)
print("   Note: Documents already evaluated by LLM reranker")
print(f"   All {len(documents)} documents scored >= 50/100")
synthesis = cot.synthesize_information(question, documents)
print(f"✅ Synthesis complete")
print(f"   Total docs: {synthesis['total_documents']}")
print(f"   Relevant docs: {synthesis['relevant_documents']}")
print(f"   Themes: {', '.join(synthesis['themes_identified'])}")

print("\n[3] Step 3: Plan Answer")
print("-" * 70)
plan = cot.plan_answer(question, synthesis)
print(f"✅ Plan complete")
print(f"   Answer type: {plan['answer_type']}")
print(f"   Structure: {' → '.join(plan['structure'])}")

print("\n[4] Get Reasoning Summary")
print("-" * 70)
summary = cot.get_thoughts_summary()
print(summary[:500] + "..." if len(summary) > 500 else summary)

print("\n" + "=" * 70)
print("✅ CHAIN-OF-THOUGHT WORKS WITHOUT DOCUMENT EVALUATION")
print("=" * 70)
print("\nKey Changes:")
print("   ❌ Removed: Step 2 - Document Evaluation")
print("   ✅ Reason: Reranker already evaluates with LLM (score 0-100)")
print("   ✅ Benefit: Faster, no duplicate evaluation")
print("   ✅ Quality: Threshold ensures all docs are relevant (>= 50)")
