"""
Chain-of-Thought (CoT) Module for RAG Pipeline
==============================================

Implements Chain-of-Thought prompting technique for improved reasoning:
1. Problem Analysis: Break down the question
2. Document Evaluation: Assess relevance of each document
3. Information Synthesis: Combine relevant information
4. Answer Construction: Build the final answer step by step
5. Validation: Verify the answer

Reference: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
Wei et al., 2022
"""

import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChainOfThought:
    """
    Chain-of-Thought reasoning for RAG pipeline
    """
    
    def __init__(self, debug: bool = True):
        """
        Initialize CoT module
        
        Args:
            debug: Enable debug printing of thoughts
        """
        self.debug = debug
        self.thoughts = []
        self.reasoning_steps = []
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Step 1: Analyze the question to understand what's being asked
        
        Args:
            question: User's question
        
        Returns:
            Analysis of question components
        """
        thought = f"🧠 STEP 1: ANALYZING QUESTION"
        self._log_thought(thought)
        
        analysis = {
            'original_question': question,
            'keywords': self._extract_keywords(question),
            'question_type': self._classify_question(question),
            'complexity_level': self._assess_complexity(question),
        }
        
        step_thought = f"""
   Question: "{question}"
   Keywords extracted: {', '.join(analysis['keywords'])}
   Question Type: {analysis['question_type']}
   Complexity: {analysis['complexity_level']}
"""
        self._log_thought(step_thought)
        self.reasoning_steps.append(('analysis', analysis))
        
        return analysis
    
    def evaluate_documents(self, 
                          question: str, 
                          documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 2: Evaluate each document's relevance to the question
        
        Args:
            question: User's question
            documents: Retrieved documents with scores
        
        Returns:
            Evaluated documents with reasoning
        """
        thought = f"🧠 STEP 2: EVALUATING DOCUMENT RELEVANCE"
        self._log_thought(thought)
        
        evaluated_docs = []
        keywords = self._extract_keywords(question)
        
        for i, doc in enumerate(documents, 1):
            doc_text = doc.get('text', '')
            
            # Calculate keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in doc_text.lower())
            match_ratio = keyword_matches / len(keywords) if keywords else 0
            
            # Assess relevance
            relevance_score = self._calculate_relevance(doc_text, question, keywords)
            
            eval_doc = {
                **doc,
                'keyword_matches': keyword_matches,
                'keyword_ratio': match_ratio,
                'relevance_score': relevance_score,
                'is_relevant': relevance_score > 0.5
            }
            
            evaluated_docs.append(eval_doc)
            
            step_thought = f"""
   Document {i}:
      Relevance Score: {relevance_score:.2f}
      Keyword Matches: {keyword_matches}/{len(keywords)}
      Is Relevant: {'✅ Yes' if relevance_score > 0.5 else '❌ No'}
      Text Preview: {doc_text[:80]}...
"""
            self._log_thought(step_thought)
        
        self.reasoning_steps.append(('document_evaluation', evaluated_docs))
        return evaluated_docs
    
    def synthesize_information(self, 
                              question: str,
                              documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 3: Synthesize information from relevant documents
        
        Args:
            question: User's question
            documents: Relevant documents
        
        Returns:
            Synthesized information summary
        """
        thought = f"🧠 STEP 3: SYNTHESIZING INFORMATION"
        self._log_thought(thought)
        
        # Group documents by topic/theme
        themes = self._extract_themes(question, documents)
        
        synthesis = {
            'question': question,
            'total_documents': len(documents),
            'relevant_documents': sum(1 for d in documents if d.get('is_relevant', False)),
            'themes_identified': themes,
            'key_points': self._extract_key_points(documents),
            'data_coverage': self._assess_data_coverage(question, documents)
        }
        
        step_thought = f"""
   Total documents reviewed: {synthesis['total_documents']}
   Relevant documents: {synthesis['relevant_documents']}
   Themes identified: {', '.join(themes)}
   Key points to address: {len(synthesis['key_points'])}
   Data coverage: {synthesis['data_coverage']}%
"""
        self._log_thought(step_thought)
        
        self.reasoning_steps.append(('synthesis', synthesis))
        return synthesis
    
    def plan_answer(self, 
                   question: str,
                   synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 4: Plan the answer structure
        
        Args:
            question: User's question
            synthesis: Synthesized information
        
        Returns:
            Answer plan with structure
        """
        thought = f"🧠 STEP 4: PLANNING ANSWER STRUCTURE"
        self._log_thought(thought)
        
        plan = {
            'answer_type': self._determine_answer_type(question),
            'structure': self._plan_structure(question, synthesis),
            'key_sections': self._identify_sections(synthesis),
            'format_recommendation': self._recommend_format(question)
        }
        
        step_thought = f"""
   Answer Type: {plan['answer_type']}
   Recommended Structure: {' → '.join(plan['structure'])}
   Key Sections: {len(plan['key_sections'])}
   Format: {plan['format_recommendation']}
"""
        self._log_thought(step_thought)
        
        self.reasoning_steps.append(('planning', plan))
        return plan
    
    def generate_cot_prompt(self,
                           question: str,
                           context: str,
                           analysis: Dict[str, Any],
                           synthesis: Dict[str, Any],
                           plan: Dict[str, Any]) -> str:
        """
        Step 5: Generate CoT-enhanced prompt for LLM
        
        Args:
            question: User's question
            context: Retrieved documents
            analysis: Question analysis
            synthesis: Synthesized information
            plan: Answer plan
        
        Returns:
            CoT-enhanced prompt
        """
        thought = f"🧠 STEP 5: GENERATING CHAIN-OF-THOUGHT PROMPT"
        self._log_thought(thought)
        
        cot_prompt = f"""You are an expert network configuration assistant. Use Chain-of-Thought reasoning to answer the question.

## QUESTION ANALYSIS
- Keywords: {', '.join(analysis['keywords'])}
- Type: {analysis['question_type']}
- Complexity: {analysis['complexity_level']}

## AVAILABLE INFORMATION
{context}

## ANSWER PLANNING
- Answer Type: {plan['answer_type']}
- Structure: {' → '.join(plan['structure'])}
- Key Topics: {', '.join(plan['key_sections'])}

## 🧠 CHAIN-OF-THOUGHT REASONING INSTRUCTIONS

**Key Prompts to Encourage Step-by-Step Thinking:**

- **Hãy suy nghĩ từng bước một.** (Think step by step.)
- **Giải thích lý do bạn đưa ra câu trả lời.** (Explain your reasoning for the answer.)
- **Suy luận từng bước.** (Reason through each step.)
- **Tại sao bạn kết luận như vậy?** (Why do you conclude that?)
- **Hãy liệt kê các bước để tìm ra câu trả lời.** (List the steps to reach the answer.)

## STEP-BY-STEP PROCESS:

1. **Understand the Question** (Hiểu câu hỏi): Confirm what is being asked
2. **Identify Relevant Information** (Xác định thông tin liên quan): Which parts of the context are relevant?
3. **Break Down the Problem** (Phân tích vấn đề): What sub-questions need to be answered?
4. **Synthesize Information** (Kết hợp thông tin): How do the relevant pieces connect?
5. **Construct Answer** (Xây dựng câu trả lời): Build the complete answer following the planned structure
6. **Validate** (Xác thực): Does the answer fully address the question?

---

Question: {question}

**Hãy suy nghĩ từng bước một. Giải thích lý do bạn đưa ra câu trả lời.**
(Let me think step by step. Explain the reasoning behind your answer.)

---

My reasoning process:

1. **Understanding the question** (Hiểu câu hỏi): 
   - Main focus: {', '.join(analysis['keywords'])}
   - What needs to be clarified: [Your analysis here]

2. **Relevant information from context** (Thông tin liên quan):
   - Key facts: [Extract relevant information]
   - Important details: [Note critical points]

3. **Breaking down the problem** (Phân tích vấn đề):
   - Sub-question 1: [Identify sub-questions]
   - Sub-question 2: [Break down complexity]
   - Dependencies: [Note how pieces relate]

4. **Synthesizing information** (Kết hợp thông tin):
   - Connection between facts: [Explain relationships]
   - Logical flow: [Describe the reasoning chain]
   - Supporting evidence: [Reference context]

5. **Constructing the answer** (Xây dựng câu trả lời):
   - Step 1: [First action/consideration]
   - Step 2: [Next action/consideration]
   - Step 3+: [Continue logically]

6. **Validation** (Xác thực):
   - Does this address: {', '.join(analysis['keywords'])}?
   - Is it consistent with the context?
   - Are all aspects covered?

---

**Final Answer:**
"""
        
        step_thought = f"""
   CoT Prompt generated with:
      - Question understanding component
      - Information identification step
      - Multi-step reasoning framework
      - Clear validation checkpoint
   Prompt length: {len(cot_prompt)} characters
"""
        self._log_thought(step_thought)
        
        return cot_prompt
    
    # Helper methods
    
    def _log_thought(self, thought: str):
        """Log a thought and add to thoughts list"""
        if self.debug:
            print(thought)
            logger.info(thought)
        self.thoughts.append(thought)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - can be enhanced
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'what', 'how', 'why', 'when', 'where'}
        words = text.lower().split()
        keywords = [w.strip('?,.:;!') for w in words if len(w) > 3 and w.lower() not in stop_words]
        return list(set(keywords))[:5]  # Return unique keywords
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'explain', 'describe']):
            return "Conceptual"
        elif any(word in question_lower for word in ['how', 'configure', 'setup', 'install']):
            return "Procedural"
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return "Causal"
        elif any(word in question_lower for word in ['compare', 'difference', 'vs']):
            return "Comparative"
        else:
            return "General"
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity"""
        word_count = len(question.split())
        
        if word_count < 5:
            return "Simple"
        elif word_count < 15:
            return "Moderate"
        else:
            return "Complex"
    
    def _calculate_relevance(self, doc_text: str, question: str, keywords: List[str]) -> float:
        """Calculate document relevance score"""
        doc_lower = doc_text.lower()
        question_lower = question.lower()
        
        # Count keyword matches
        keyword_score = sum(1 for kw in keywords if kw in doc_lower) / len(keywords) if keywords else 0
        
        # Check for exact phrase matches
        phrase_score = 1.0 if question_lower in doc_lower else 0.0
        
        # Length appropriateness (prefer medium-length docs)
        length_score = min(1.0, len(doc_text) / 500)  # Normalize by 500 chars
        
        # Combined score
        relevance = (keyword_score * 0.6) + (phrase_score * 0.3) + (length_score * 0.1)
        return min(1.0, relevance)
    
    def _extract_themes(self, question: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from documents"""
        themes = []
        
        # Simple theme extraction - can be enhanced
        if any(word in question.lower() for word in ['config', 'command', 'setup']):
            themes.append("Configuration")
        if any(word in question.lower() for word in ['ospf', 'bgp', 'eigrp', 'routing']):
            themes.append("Routing")
        if any(word in question.lower() for word in ['interface', 'port', 'ethernet']):
            themes.append("Interfaces")
        if any(word in question.lower() for word in ['acl', 'security', 'authentication']):
            themes.append("Security")
        
        return themes if themes else ["General"]
    
    def _extract_key_points(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from documents"""
        key_points = []
        
        for doc in documents:
            if doc.get('is_relevant'):
                text = doc.get('text', '')
                # Extract first sentence as key point
                sentence = text.split('.')[0] if '.' in text else text[:100]
                key_points.append(sentence.strip())
        
        return key_points[:3]  # Return top 3 key points
    
    def _assess_data_coverage(self, question: str, documents: List[Dict[str, Any]]) -> int:
        """Assess how well documents cover the question"""
        if not documents:
            return 0
        
        relevant_count = sum(1 for d in documents if d.get('is_relevant', False))
        coverage = int((relevant_count / len(documents)) * 100) if documents else 0
        
        return coverage
    
    def _determine_answer_type(self, question: str) -> str:
        """Determine what type of answer is expected"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['code', 'command', 'configure', 'script']):
            return "Code/Configuration"
        elif any(word in question_lower for word in ['explain', 'what', 'define']):
            return "Explanation"
        elif any(word in question_lower for word in ['steps', 'process', 'procedure', 'how']):
            return "Procedure"
        else:
            return "Information"
    
    def _plan_structure(self, question: str, synthesis: Dict[str, Any]) -> List[str]:
        """Plan the structure of the answer"""
        structure = []
        
        answer_type = self._determine_answer_type(question)
        
        if "Code" in answer_type:
            structure = ["Overview", "Prerequisites", "Code Example", "Explanation", "Validation"]
        elif "Explanation" in answer_type:
            structure = ["Definition", "Key Concepts", "Examples", "Related Topics"]
        elif "Procedure" in answer_type:
            structure = ["Prerequisites", "Step-by-step", "Verification", "Troubleshooting"]
        else:
            structure = ["Summary", "Details", "Examples", "Additional Info"]
        
        return structure
    
    def _identify_sections(self, synthesis: Dict[str, Any]) -> List[str]:
        """Identify key sections to include"""
        return synthesis.get('themes_identified', [])
    
    def _recommend_format(self, question: str) -> str:
        """Recommend output format"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['code', 'command', 'config']):
            return "Code blocks with explanation"
        elif any(word in question_lower for word in ['step', 'procedure', 'how']):
            return "Numbered steps"
        else:
            return "Paragraph with examples"
    
    def get_thoughts_summary(self) -> str:
        """Get summary of all thoughts"""
        return "\n".join(self.thoughts)
    
    def get_reasoning_trace(self) -> List[tuple]:
        """Get the complete reasoning trace"""
        return self.reasoning_steps
    
    def reset(self):
        """Reset thoughts and reasoning steps"""
        self.thoughts = []
        self.reasoning_steps = []
