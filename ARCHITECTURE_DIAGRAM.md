# Chain-of-Thought Reasoning Prompts - Architecture Diagram

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER QUESTION                               │
│              "Configure OSPF on router R1"                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │    RAG PIPELINE (rag/pipeline.py)  │
        │   enable_cot=True (Auto-enabled)   │
        └────────┬─────────────────┬─────────┘
                 │                 │
    Step 1:      │   Step 2:       │   Step 3:      Step 4:
    Retrieve     │   Rerank        │   CoT          Generate
    ▼            ▼                 ▼                ▼
┌──────────┐ ┌──────────┐  ┌──────────────────────────────────┐
│ Retriever│─→│Reranker  │──→│ CHAIN-OF-THOUGHT MODULE        │
│(top-10)  │  │(top-5)   │  │ (rag/chain_of_thought.py)      │
└──────────┘  └──────────┘  │                                 │
                             │ 🧠 Analyze Question            │
                             │    Synthesize Information      │
                             │    Plan Answer Structure       │
                             │    Generate CoT Prompt         │
                             └──────────────┬──────────────────┘
                                           │
                          ┌─────────────────┴──────────────┐
                          │                                 │
                    CoT Prompt Generated                   │
                          │                                 │
                    ┌─────▼─────────────────────────────┐  │
                    │  REASONING PROMPTS ADDED:         │  │
                    │                                  │  │
                    │ ✓ Hãy suy nghĩ từng bước một.   │  │
                    │   (Think step by step)            │  │
                    │                                  │  │
                    │ ✓ Giải thích lý do bạn           │  │
                    │   đưa ra câu trả lời.            │  │
                    │   (Explain your reasoning)       │  │
                    │                                  │  │
                    │ ✓ Suy luận từng bước.            │  │
                    │   (Reason through each step)     │  │
                    │                                  │  │
                    │ ✓ Tại sao bạn kết luận           │  │
                    │   như vậy?                       │  │
                    │   (Why do you conclude that?)    │  │
                    │                                  │  │
                    │ ✓ Hãy liệt kê các bước           │  │
                    │   để tìm ra câu trả lời.         │  │
                    │   (List steps to the answer)     │  │
                    └──────────┬───────────────────────┘  │
                               │                           │
                               ▼                           │
                    ┌─────────────────────────┐           │
                    │  LLM CLIENT             │           │
                    │ (rag/llm_client.py)    │◄──────────┘
                    │                        │
                    │ _build_prompt():      │
                    │  • System prompt       │
                    │  • Reasoning section  │
                    │  • 5-step framework   │
                    │  • Question + Context│
                    │  • Thinking prompt    │
                    └─────────┬─────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  OLLAMA API        │
                    │ (qwen2.5-coder:3b) │
                    └──────────┬─────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │   LLM GENERATES ANSWER WITH          │
            │   STEP-BY-STEP THINKING              │
            │                                      │
            │ Output includes:                    │
            │ ✓ Question Analysis                │
            │ ✓ Information Identification       │
            │ ✓ Step-by-Step Reasoning          │
            │ ✓ Explanation of Logic            │
            │ ✓ Constructed Answer              │
            │ ✓ Validation Verification         │
            └──────────────┬───────────────────┘
                           │
                           ▼
           ┌───────────────────────────────┐
           │    FINAL ANSWER               │
           │                               │
           │ Let me think step by step:    │
           │                               │
           │ 1. Understanding: Configure  │
           │    OSPF on R1...             │
           │                               │
           │ 2. Relevant info: Process ID,│
           │    network definition...     │
           │                               │
           │ 3. Breaking down: Enter      │
           │    config → Create process   │
           │    → Define networks...      │
           │                               │
           │ 4. Synthesizing: Order       │
           │    matters...                │
           │                               │
           │ 5. Answer:                   │
           │    R1#configure              │
           │    R1(config)#router...      │
           │                               │
           │ 6. Validation: ✓ Complete   │
           └───────────────────────────────┘
```

---

## 🔄 Data Flow with Reasoning

```
Question Input
     │
     ├─→ RAG Pipeline
     │   │
     │   ├─→ Retrieve: Top-10 documents
     │   │
     │   ├─→ Rerank: Top-5 documents
     │   │
     │   └─→ CoT Analysis
     │       │
     │       ├─→ analyze_question()
     │       ├─→ evaluate_documents()
     │       ├─→ synthesize_information()
     │       ├─→ plan_answer()
     │       │
     │       └─→ generate_cot_prompt()
     │           │
     │           ├─ Add reasoning_prompt section
     │           ├─ Add 5 Vietnamese prompts
     │           ├─ Add step-by-step framework
     │           └─ Format as complete prompt
     │
     └─→ LLM Client (llm_client.py)
         │
         ├─→ _build_prompt()
         │   │
         │   ├─ System prompt
         │   ├─ REASONING INSTRUCTIONS
         │   │  (from earlier step)
         │   ├─ Context
         │   ├─ Question
         │   └─ "Hãy suy nghĩ từng bước một."
         │
         └─→ Call Ollama API
             │
             └─→ LLM thinks and responds
                 with step-by-step reasoning
```

---

## 5️⃣ Five-Step Thinking Framework

```
┌─────────────────────────────────────────────────┐
│  5-STEP THINKING FRAMEWORK (Vietnamese)          │
└─────────────────────────────────────────────────┘

STEP 1: Phân tích câu hỏi
├─ Analyze the question
├─ Break down what's being asked
├─ Extract keywords
└─ Identify question type

        │
        ▼

STEP 2: Xác định thông tin cần thiết
├─ Identify necessary information
├─ Find relevant documents
├─ Extract key facts
└─ List requirements

        │
        ▼

STEP 3: Suy luận từng bước
├─ Reason step by step
├─ Break down complex problems
├─ Show intermediate steps
└─ Connect information logically

        │
        ▼

STEP 4: Giải thích lý do
├─ Explain your reasoning
├─ Justify conclusions
├─ Show logic flow
└─ Validate reasoning

        │
        ▼

STEP 5: Xây dựng câu trả lời
├─ Construct the answer
├─ Build final response
├─ Include step-by-step solution
└─ Provide verification
```

---

## 🎯 Reasoning Prompts Map

```
┌────────────────────────────────────────────────────┐
│          REASONING PROMPTS APPLIED                  │
└────────────────────────────────────────────────────┘

In: rag/llm_client.py._build_prompt()
│
├─ Prompt Part 1: System Instructions
├─ Prompt Part 2: REASONING SECTION (NEW)
│  │
│  ├─ "Hãy suy nghĩ từng bước một."
│  │   └─ In llm_client.py (all calls)
│  │
│  ├─ "Giải thích lý do bạn đưa ra..."
│  │   └─ In chain_of_thought.py (CoT calls)
│  │
│  ├─ "Suy luận từng bước."
│  │   └─ In llm_client.py (all calls)
│  │
│  ├─ "Tại sao bạn kết luận như vậy?"
│  │   └─ In chain_of_thought.py (CoT calls)
│  │
│  └─ "Hãy liệt kê các bước..."
│      └─ In chain_of_thought.py (CoT calls)
│
├─ Prompt Part 3: 5-Step Framework
├─ Prompt Part 4: Context
├─ Prompt Part 5: Question
└─ Prompt Part 6: Thinking Trigger
   └─ "Hãy suy nghĩ từng bước một."

        │
        ▼

Out: LLM generates step-by-step response
     with explicit reasoning shown
```

---

## 📊 Execution Flow with Timing

```
User Question (t=0)
    │
    ▼
Retrieve Documents (t≈0.5s)
    │
    ▼
Rerank Documents (t≈5-10s)
    │
    ▼
CoT Analysis (t≈1-2s)
    │ - Analyze question
    │ - Synthesize info
    │ - Plan structure
    │
    ▼
LLM Client _build_prompt() (t≈0.1s)
    │ - Add reasoning_prompt section
    │ - Format all sections
    │ - Ready to send to LLM
    │
    ▼
Call Ollama API (t≈30-120s)
    │ - Send prompt with reasoning
    │ - LLM processes thinking prompts
    │ - LLM generates step-by-step answer
    │
    ▼
Return Result (t≈30-120s)
    │ - Answer with reasoning trace
    │ - Timestamps and token counts
    │ - Cached if applicable
    │
    ▼
Display to User (t≈0.01s)
    └─ Complete step-by-step response
      with reasoning shown

Total Time: ~30-150 seconds
```

---

## 🧩 Component Integration

```
┌─────────────────────────────────────────────────┐
│         COMPONENT INTEGRATION MAP                 │
└─────────────────────────────────────────────────┘

main.py
  │
  └─→ RAG Pipeline (rag/pipeline.py)
      │
      ├─→ Retriever (agent/retriever.py)
      │
      ├─→ Reranker (agent/reranker.py)
      │
      ├─→ Chain-of-Thought (rag/chain_of_thought.py)
      │   ├─ Reasoning prompts added here ✓
      │   └─ Calls generate_cot_prompt()
      │
      └─→ LLM Client (rag/llm_client.py)
          ├─ Reasoning prompts added here ✓
          ├─ Calls _build_prompt()
          └─→ Call Ollama API
              └─→ Response with reasoning

Files Modified:
  • rag/llm_client.py
  • rag/chain_of_thought.py

Files Created:
  • test_cot_reasoning_prompts.py
  • COT_REASONING_PROMPTS.md
  • COT_REASONING_IMPLEMENTATION.md
  • QUICK_REFERENCE_COT.md
  • COT_SETUP_COMPLETE.md
  • IMPLEMENTATION_SUMMARY.md
```

---

## ✅ Verification Checklist

```
✅ Reasoning Prompts Added
   └─ Vietnamese: ✓ (5 prompts)
   └─ English: ✓ (5 translations)

✅ LLM Client Enhanced
   └─ rag/llm_client.py: ✓
   └─ _build_prompt() modified: ✓

✅ Chain-of-Thought Enhanced
   └─ rag/chain_of_thought.py: ✓
   └─ generate_cot_prompt() modified: ✓

✅ Integration Complete
   └─ All pipeline queries: ✓
   └─ All CoT prompts: ✓

✅ Tests Created
   └─ 3 test cases: ✓
   └─ All passing: ✓

✅ Documentation
   └─ 4 doc files: ✓
   └─ Complete guides: ✓

✅ Production Ready
   └─ No breaking changes: ✓
   └─ Backward compatible: ✓
   └─ Ready to deploy: ✓
```

---

## 🚀 Usage Quick Start

```
1. Use Pipeline (Default, Recommended):
   pipeline = RAGPipeline(enable_cot=True)
   result = pipeline.query("Your question")
   
   ✓ Reasoning prompts automatic
   ✓ Step-by-step thinking shown
   ✓ No code changes needed

2. Use LLM Client:
   llm = LLMClient()
   result = llm.generate(query="Q", context="C")
   
   ✓ Reasoning prompts included
   ✓ All LLM calls enhanced
   ✓ Transparent reasoning

3. Use CoT Directly:
   cot = ChainOfThought(debug=True)
   cot_prompt = cot.generate_cot_prompt(...)
   result = llm.generate(use_cot=True, cot_prompt=cot_prompt)
   
   ✓ Full control of reasoning
   ✓ Detailed thinking trace
   ✓ Complete transparency
```

---

**🎉 System is fully integrated and ready for use!**
