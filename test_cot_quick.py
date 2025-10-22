#!/usr/bin/env python3
"""
Quick Chain-of-Thought Test
============================

Demonstrates CoT reasoning without waiting for LLM generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.chain_of_thought import ChainOfThought


def demo_cot_reasoning():
    """Demonstrate Chain-of-Thought reasoning steps"""
    
    print("\n" + "=" * 80)
    print("🧠 CHAIN-OF-THOUGHT REASONING DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Initialize CoT with debug enabled
    cot = ChainOfThought(debug=True)
    
    # Example question
    question = "How do I configure OSPF routing protocol on ZebOS with BFD for fast failover?"
    
    print("📝 Question to analyze:")
    print(f'   "{question}"\n')
    
    # Step 1: Analyze Question
    print("\n" + "-" * 80)
    print("STEP 1: ANALYZING THE QUESTION")
    print("-" * 80)
    analysis = cot.analyze_question(question)
    
    # Step 2: Evaluate Documents
    print("\n" + "-" * 80)
    print("STEP 2: EVALUATING DOCUMENT RELEVANCE")
    print("-" * 80)
    
    # Mock retrieved documents
    mock_documents = [
        {
            'text': 'OSPF (Open Shortest Path First) is an interior gateway routing protocol used in IP networks. It uses the Dijkstra algorithm to calculate the shortest path. OSPF supports BFD (Bidirectional Forwarding Detection) for fast failure detection and recovery.',
            'score': 0.85,
            'id': 1
        },
        {
            'text': 'To configure OSPF on ZebOS, use the following commands: configure terminal, router ospf 1, network 10.0.0.0 0.0.0.255 area 0',
            'score': 0.90,
            'id': 2
        },
        {
            'text': 'BGP (Border Gateway Protocol) is used for routing between autonomous systems on the internet.',
            'score': 0.40,
            'id': 3
        },
        {
            'text': 'RIP is an old routing protocol rarely used in modern networks.',
            'score': 0.10,
            'id': 4
        },
        {
            'text': 'BFD (Bidirectional Forwarding Detection) enables rapid detection of link failures by exchanging keepalive messages at subsecond intervals.',
            'score': 0.75,
            'id': 5
        }
    ]
    
    evaluated_docs = cot.evaluate_documents(question, mock_documents)
    
    # Step 3: Synthesize Information
    print("\n" + "-" * 80)
    print("STEP 3: SYNTHESIZING INFORMATION")
    print("-" * 80)
    
    synthesis = cot.synthesize_information(question, evaluated_docs)
    
    # Step 4: Plan Answer
    print("\n" + "-" * 80)
    print("STEP 4: PLANNING ANSWER STRUCTURE")
    print("-" * 80)
    
    plan = cot.plan_answer(question, synthesis)
    
    # Step 5: Generate CoT Prompt
    print("\n" + "-" * 80)
    print("STEP 5: GENERATING CHAIN-OF-THOUGHT PROMPT")
    print("-" * 80)
    
    context = "\n".join([f"[{i}] {doc['text']}" for i, doc in enumerate(evaluated_docs, 1)])
    cot_prompt = cot.generate_cot_prompt(question, context, analysis, synthesis, plan)
    
    # Display the final CoT prompt
    print("\n" + "=" * 80)
    print("📋 GENERATED CHAIN-OF-THOUGHT PROMPT FOR LLM")
    print("=" * 80 + "\n")
    print(cot_prompt)
    print("\n" + "=" * 80)
    
    # Show reasoning trace summary
    print("\n" + "=" * 80)
    print("📊 REASONING TRACE SUMMARY")
    print("=" * 80)
    
    reasoning_trace = cot.get_reasoning_trace()
    for step_name, step_data in reasoning_trace:
        print(f"\n✓ {step_name.upper().replace('_', ' ')}")
        if isinstance(step_data, dict):
            for key, value in step_data.items():
                if isinstance(value, list):
                    print(f"  └─ {key}: {len(value)} items")
                else:
                    print(f"  └─ {key}: {value}")
        elif isinstance(step_data, list):
            print(f"  └─ {len(step_data)} items analyzed")
    
    # Show all thoughts
    print("\n" + "=" * 80)
    print("💭 COMPLETE THOUGHT PROCESS")
    print("=" * 80)
    print(cot.get_thoughts_summary())


def compare_question_types():
    """Compare how CoT handles different types of questions"""
    
    print("\n" + "=" * 80)
    print("🔍 ANALYZING DIFFERENT QUESTION TYPES WITH CoT")
    print("=" * 80 + "\n")
    
    questions = [
        "What is OSPF?",  # Conceptual
        "How do I configure OSPF on a ZebOS router?",  # Procedural
        "Why should I use OSPF instead of BGP in my network?",  # Causal
        "What are the differences between OSPF and EIGRP?",  # Comparative
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print('='*80)
        
        cot = ChainOfThought(debug=False)  # Quiet mode
        analysis = cot.analyze_question(question)
        
        print(f"Question Type: {analysis['question_type']}")
        print(f"Complexity: {analysis['complexity_level']}")
        print(f"Keywords: {', '.join(analysis['keywords'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Chain-of-Thought Test")
    parser.add_argument('--types', action='store_true', help='Compare question types')
    args = parser.parse_args()
    
    if args.types:
        compare_question_types()
    else:
        demo_cot_reasoning()
