#!/usr/bin/env python3
"""
Quick verification that paths are fixed
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("PATH FIX VERIFICATION")
print("=" * 70)

# Test 1: Import retriever
print("\n[1] Testing HybridRetriever import and initialization...")
try:
    from agent.retriever import HybridRetriever
    retriever = HybridRetriever()
    print(f"✅ HybridRetriever initialized successfully")
    print(f"   Loaded {len(retriever.texts)} text chunks")
except Exception as e:
    print(f"❌ HybridRetriever failed: {e}")
    sys.exit(1)

# Test 2: Import multi-index retriever
print("\n[2] Testing MultiIndexRetriever...")
try:
    from agent.multi_index_retriever import MultiIndexRetriever
    multi_retriever = MultiIndexRetriever()
    print(f"✅ MultiIndexRetriever initialized successfully")
    if multi_retriever.main_retriever:
        print(f"   Main retriever: {len(multi_retriever.main_retriever.texts)} docs")
    if multi_retriever.commands_index:
        print(f"   Commands retriever: {multi_retriever.commands_index.ntotal} vectors")
except Exception as e:
    print(f"❌ MultiIndexRetriever failed: {e}")
    sys.exit(1)

# Test 3: Test a simple query
print("\n[3] Testing simple query...")
try:
    result = retriever.retrieve("OSPF configuration", top_k=3)
    print(f"✅ Query successful, retrieved {len(result)} documents")
    if result:
        print(f"   Top result score: {result[0].get('score', 0):.4f}")
except Exception as e:
    print(f"❌ Query failed: {e}")
    sys.exit(1)

# Test 4: Check database paths
print("\n[4] Verifying database paths...")
from agent.retriever import FAISS_INDEX_PATH, METADATA_PATH, TFIDF_PATH, SVD_PATH
from pathlib import Path

paths = {
    "FAISS Index": FAISS_INDEX_PATH,
    "Metadata": METADATA_PATH,
    "TF-IDF": TFIDF_PATH,
    "SVD": SVD_PATH
}

all_exist = True
for name, path in paths.items():
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ Some database files are missing!")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - PATH FIX SUCCESSFUL!")
print("=" * 70)
print("\nThe retriever can now find database files correctly.")
print("You can run: python3 test_two_stage_pipeline.py")
