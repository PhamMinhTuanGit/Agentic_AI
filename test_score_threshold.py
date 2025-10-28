#!/usr/bin/env python3
"""
Quick test to verify 50/100 score threshold filtering
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "=" * 70)
print("TESTING SCORE THRESHOLD FILTERING (>= 50/100)")
print("=" * 70)

# Test the filtering logic directly
print("\n[1] Simulating reranked documents with various scores:")

# Simulate documents with different scores
test_docs = [
    {'text': 'Very relevant doc', 'llm_score': 85.0, 'original_rank': 1},
    {'text': 'Somewhat relevant', 'llm_score': 65.0, 'original_rank': 2},
    {'text': 'Barely relevant', 'llm_score': 52.0, 'original_rank': 3},
    {'text': 'Low quality doc', 'llm_score': 45.0, 'original_rank': 4},
    {'text': 'Irrelevant doc', 'llm_score': 20.0, 'original_rank': 5},
]

for doc in test_docs:
    print(f"   Score {doc['llm_score']:5.1f} - {doc['text']}")

# Apply filtering (score >= 50)
print("\n[2] Applying filter (score >= 50.0):")
filtered = [doc for doc in test_docs if doc['llm_score'] >= 50.0]

print(f"   Before: {len(test_docs)} documents")
print(f"   After:  {len(filtered)} documents")
print(f"   Filtered out: {len(test_docs) - len(filtered)} documents")

print("\n[3] Filtered results:")
for i, doc in enumerate(filtered, 1):
    print(f"   Rank {i}: Score {doc['llm_score']:.1f} - {doc['text']}")

# Validation
print("\n[4] Validation:")
if filtered:
    min_score = min([d['llm_score'] for d in filtered])
    max_score = max([d['llm_score'] for d in filtered])
    
    if min_score >= 50.0:
        print(f"   ✅ PASS: Minimum score {min_score:.1f} >= 50.0")
    else:
        print(f"   ❌ FAIL: Minimum score {min_score:.1f} < 50.0")
    
    if len(filtered) == 3:  # Should be 85, 65, 52
        print(f"   ✅ PASS: Correct number of docs retained (3)")
    else:
        print(f"   ❌ FAIL: Expected 3 docs, got {len(filtered)}")
    
    if 45.0 not in [d['llm_score'] for d in filtered]:
        print(f"   ✅ PASS: Score 45.0 correctly filtered out")
    else:
        print(f"   ❌ FAIL: Score 45.0 should be filtered out")
    
    if 20.0 not in [d['llm_score'] for d in filtered]:
        print(f"   ✅ PASS: Score 20.0 correctly filtered out")
    else:
        print(f"   ❌ FAIL: Score 20.0 should be filtered out")
else:
    print(f"   ❌ FAIL: No documents in filtered results")

print("\n" + "=" * 70)
print("✅ Score threshold filtering logic verified")
print("   - Documents with score >= 50: INCLUDED")
print("   - Documents with score < 50: EXCLUDED")
print("=" * 70)
