#!/usr/bin/env python3
"""
Test Chain-of-Thought with Reasoning Prompts
=============================================

Demonstrates how reasoning prompts encourage step-by-step thinking:
- "Hãy suy nghĩ từng bước một." (Think step by step)
- "Giải thích lý do bạn đưa ra câu trả lời." (Explain your reasoning)
- "Suy luận từng bước." (Reason through each step)
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.llm_client import LLMClient
from rag.chain_of_thought import ChainOfThought

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_reasoning_prompts():
    """Test LLM reasoning prompts"""
    
    print("\n" + "=" * 80)
    print("🧠 CHAIN-OF-THOUGHT REASONING PROMPTS TEST")
    print("=" * 80)
    
    # Initialize LLM client
    print("\n✅ Initializing LLM Client...")
    llm = LLMClient(
        model="qwen2.5-coder:3b",
        temperature=0.3,  # Lower temp for more deterministic reasoning
        max_tokens=1024
    )
    
    # Test 1: Simple reasoning question
    print("\n" + "-" * 80)
    print("TEST 1: Simple Reasoning Question")
    print("-" * 80)
    
    query1 = "If a router takes 2 hours to traverse a link with speed 100 Mbps, how many Megabits can it send?"
    context1 = """
    Router speed: 100 Mbps
    Duration: 2 hours = 7200 seconds
    Formula: Data = Speed × Time
    """
    
    print(f"\n📝 Question: {query1}")
    print(f"📋 Context: {context1.strip()}")
    print("\n🔄 Generating answer with reasoning prompts...\n")
    
    result1 = llm.generate(
        query=query1,
        context=context1,
        output_format="default",
        session_type="general"
    )
    
    print("=" * 80)
    print("💬 ANSWER WITH REASONING:")
    print("=" * 80)
    print(result1['answer'])
    print("=" * 80)
    print(f"⏱️  Time: {result1['elapsed_time']:.2f}s | Tokens: {result1['total_tokens']}")
    
    # Test 2: Network configuration reasoning
    print("\n" + "-" * 80)
    print("TEST 2: Network Configuration Reasoning")
    print("-" * 80)
    
    query2 = "What are the steps to configure OSPF on a router?"
    context2 = """
    OSPF Configuration:
    1. Enter configuration mode: configure
    2. Create OSPF process: router ospf <process_id>
    3. Define networks: network <ip> <wildcard> area <area_id>
    4. Configure interfaces: interface <name> then ospf cost
    5. Enable OSPF: no shutdown
    """
    
    print(f"\n📝 Question: {query2}")
    print(f"📋 Context: {context2.strip()}")
    print("\n🔄 Generating answer with reasoning prompts...\n")
    
    result2 = llm.generate(
        query=query2,
        context=context2,
        output_format="single_code_block",
        session_type="general"
    )
    
    print("=" * 80)
    print("💬 ANSWER WITH REASONING:")
    print("=" * 80)
    print(result2['answer'])
    print("=" * 80)
    print(f"⏱️  Time: {result2['elapsed_time']:.2f}s | Tokens: {result2['total_tokens']}")
    
    # Test 3: Chain-of-Thought detailed reasoning
    print("\n" + "-" * 80)
    print("TEST 3: Chain-of-Thought Detailed Reasoning")
    print("-" * 80)
    
    print("\n✅ Initializing Chain-of-Thought module...")
    cot = ChainOfThought(debug=True)
    
    query3 = "Configure OSPF on router R1 for network 10.0.0.0/8"
    context3 = """
    Router Configuration Best Practices:
    - Always plan before configuring
    - Test routing in lab first
    - Document all changes
    - OSPF requires network planning
    
    OSPF Setup Requirements:
    - IP addresses on interfaces
    - Process ID selection
    - Area definitions
    - Router ID configuration
    """
    
    print(f"\n📝 Question: {query3}")
    print(f"📋 Context: {context3.strip()}")
    
    # Generate CoT steps
    print("\n🧠 Generating Chain-of-Thought analysis...\n")
    analysis = cot.analyze_question(query3)
    synthesis = cot.synthesize_information(query3, [{'text': context3}])
    plan = cot.plan_answer(query3, synthesis)
    
    # Generate CoT prompt
    cot_prompt = cot.generate_cot_prompt(query3, context3, analysis, synthesis, plan)
    
    print("\n" + "=" * 80)
    print("📋 CHAIN-OF-THOUGHT PROMPT (with reasoning triggers):")
    print("=" * 80)
    print(cot_prompt[:1500])  # Show first 1500 chars
    if len(cot_prompt) > 1500:
        print(f"... [+{len(cot_prompt) - 1500} more characters]")
    print("=" * 80)
    
    # Use the CoT prompt with LLM
    print("\n🔄 Generating answer using CoT prompt with reasoning...\n")
    
    result3 = llm.generate(
        query=query3,
        context=context3,
        use_cot=True,
        cot_prompt=cot_prompt
    )
    
    print("=" * 80)
    print("💬 ANSWER WITH CHAIN-OF-THOUGHT REASONING:")
    print("=" * 80)
    print(result3['answer'])
    print("=" * 80)
    print(f"⏱️  Time: {result3['elapsed_time']:.2f}s | Tokens: {result3['total_tokens']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print("""
✅ Reasoning Prompts Applied:
   - "Hãy suy nghĩ từng bước một." (Think step by step)
   - "Giải thích lý do bạn đưa ra câu trả lời." (Explain your reasoning)
   - "Suy luận từng bước." (Reason through each step)

📈 These prompts encourage the LLM to:
   1. Break down the problem step-by-step
   2. Show intermediate reasoning steps
   3. Explain the logical connection between steps
   4. Provide a comprehensive reasoning trace
   5. Validate the answer against the question

🎯 Expected Behavior:
   - More detailed explanations
   - Step-by-step problem solving
   - Better accuracy through explicit reasoning
   - Clearer logic flow in answers
    """)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_reasoning_prompts()
