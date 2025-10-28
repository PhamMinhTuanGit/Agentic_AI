import os
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
RERANK_API_URL = os.getenv("RERANK_API_URL", "http://localhost:11434/api/generate")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen2.5-coder:3b")
RERANK_TIMEOUT = int(os.getenv("RERANK_TIMEOUT", "120"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMReranker:
    """
    LLM-based reranker using Qwen2.5-Coder for relevance scoring
    """
    
    def __init__(self, 
                 model: str = RERANK_MODEL,
                 api_url: str = LLM_API_URL,
                 top_k: int = 5,
                 temperature: float = 0.1,
                 timeout: int = 60):
        """
        Initialize LLM Reranker
        
        Args:
            model: LLM model name (e.g., 'qwen2.5-coder:3b')
            api_url: Ollama API endpoint
            top_k: Number of documents to return after reranking
            temperature: Temperature for LLM (lower = more deterministic)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.api_url = api_url
        self.top_k = top_k
        self.temperature = temperature
        self.timeout = timeout
        
        logger.info(f"✅ Initialized LLM Reranker with model: {model}")
    
    def _create_rerank_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Create a comprehensive prompt for RAG-based document reranking
        
        Uses a detailed evaluation framework to assess document relevance
        for retrieval augmented generation systems.
        
        Args:
            query: Original query
            documents: List of documents with text and rank
        
        Returns:
            Prompt string for the reranker model
        """
        # Build passages section
        passages_text = ""
        for i, doc in enumerate(documents):
            # Use full text if available, otherwise truncate
            text = doc.get('text', '')
            if len(text) > 500:
                text = text[:500] + "..."
            passages_text += f"<passage id='id{i}'>{text}</passage>\n"
        
        # Create comprehensive RAG reranking prompt
        prompt = f"""You are a customer support answer service. Your task is to evaluate help center passages and score their relevance to a given customer query for a retrieval augmented generation (RAG) system.

Evaluation Process:
1. Analyze the customer's query to identify both explicit needs and implicit context including underlying user goals
2. Assess each passage's ability to directly resolve the query or provide substantive supporting information with actionable guidance
3. Score based on how effectively the passage addresses the query's core intent while considering potential interpretations

Grading Criteria:
<grading_scale>
10: EXCEPTIONAL match - Contains exact step-by-step instructions that perfectly match the query's specific scenario. Must include all required parameters/context and resolve the issue completely without any ambiguity. Reserved for definitive solutions that exactly mirror the user's described situation and require no interpretation.

9: NEAR-PERFECT solution - Contains all critical steps for resolution but may lack one minor non-essential detail. Addresses the precise query parameters with specialized information. Solution must be directly applicable without requiring adaptation or assumptions.

8: STRONG MATCH - Provides complete technical resolution through specific instructions, but may require simple logical inferences for full application. Covers all essential components but might need minor contextualization.

7: GOOD MATCH - Contains substantial relevant details that address core aspects of the query, but lacks one important element for complete resolution. Provides concrete guidance requiring some user interpretation.

6: PARTIAL match – General guidance on the right topic but lacks the specifics for direct application. May only resolve a subset of the request.

5: LIMITED relevance – Related context or approach, but indirect. Requires substantial effort to adapt to the user's exact need.

4: TANGENTIAL – Mentions related concepts/keywords with little practical connection to the request. Minimal actionable value.

3: VAGUE domain info – Talks about the general area but not the query's specifics. No concrete, actionable steps.

2: TOKEN overlap – Shares isolated terms without context or intent aligned to the request. Similarity is coincidental.

1: IRRELEVANT – Uses query terms in a completely unrelated way. No meaningful link to the user's goal.

0: UNRELATED – No thematic or contextual connection to the query at all.
</grading_scale>

Input Format:
<input_format>
<query>
{query}
</query>
<passages>
{passages_text}</passages>
</input_format>

Output Format:
<output_format>
Return your response in a valid JSON (skip spaces):
{{"id0":score0,"id1":score1,...}}

Strict guidelines:
- Return ONLY a well-formed valid JSON with passage IDs as keys
- Each key must be a passage id in the format "idN"
- Each score must be an integer between 5 to 10. EXCLUDE passages that score below 5 (i.e. 0, 1, 2, 3 or 4)
- Integer values only, no decimals
- Skip spaces in the JSON
- No additional text or formatting
- Maintain original passage ID order
- Note: If NO passages score 5+, return empty JSON object {{}}
</output_format>

Now evaluate the passages and return ONLY the JSON:"""
        
        return prompt
    
    def _parse_scores(self, response_text: str, num_documents: int) -> Optional[List[float]]:
        """
        Parse LLM response to extract scores from JSON object format
        
        Expected format: {"id0":8,"id1":6,"id2":9,...}
        
        Args:
            response_text: LLM response
            num_documents: Expected number of documents
        
        Returns:
            List of scores (in order) or None if parsing fails
        """
        try:
            # Clean response
            response_text = response_text.strip()
            
            # Find JSON object (first try curly braces)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                # Fallback: try to find array format (backward compatibility)
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx == -1 or end_idx == 0:
                    logger.warning(f"⚠️ No JSON found in response")
                    logger.debug(f"Response: {response_text[:200]}")
                    return None
            
            json_str = response_text[start_idx:end_idx]
            
            # Parse JSON
            import json
            parsed = json.loads(json_str)
            
            # Handle object format: {"id0":8, "id1":6, ...}
            if isinstance(parsed, dict):
                # Extract scores in order (id0, id1, id2, ...)
                scores = []
                for i in range(num_documents):
                    key = f"id{i}"
                    if key in parsed:
                        score = float(parsed[key])
                        # Convert 0-10 scale to 0-100 scale
                        if score <= 10:
                            score = score * 10
                        scores.append(score)
                    else:
                        # If passage not in response, it scored below 5
                        # Assign minimum score (0)
                        scores.append(0.0)
                
                logger.debug(f"✅ Parsed {len(scores)} scores from object format")
                return scores
            
            # Handle array format (backward compatibility): [85, 45, 92, ...]
            elif isinstance(parsed, list):
                scores = [float(s) for s in parsed]
                
                # Validate count
                if len(scores) != num_documents:
                    logger.warning(f"⚠️ Score count mismatch: expected {num_documents}, got {len(scores)}")
                    # Pad or truncate
                    if len(scores) < num_documents:
                        scores.extend([0.0] * (num_documents - len(scores)))
                    else:
                        scores = scores[:num_documents]
                
                logger.debug(f"✅ Parsed {len(scores)} scores from array format")
                return scores
            
            else:
                logger.warning(f"⚠️ Unexpected JSON type: {type(parsed)}")
                return None
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            logger.debug(f"JSON string: {json_str[:200] if 'json_str' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing scores: {e}")
            logger.debug(f"Response: {response_text[:200]}")
            return None
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Call Ollama LLM API
        
        Args:
            prompt: Prompt to send to LLM
        
        Returns:
            LLM response or None if error
        """
        try:
            logger.debug(f"🔄 Calling LLM: {self.model}")
            logger.debug(f"   API: {self.api_url}")
            logger.debug(f"   Prompt length: {len(prompt)} chars")
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=self.timeout
            )
            
            # Log response status
            logger.debug(f"   Response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"❌ LLM API returned {response.status_code}")
                logger.error(f"   Response: {response.text[:200]}")
                return None
            
            result = response.json()
            llm_response = result.get("response", "")
            logger.debug(f"   Response length: {len(llm_response)} chars")
            return llm_response
        
        except requests.exceptions.Timeout:
            logger.error(f"❌ LLM request timeout ({self.timeout}s)")
            logger.error(f"   Model: {self.model}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Failed to connect to LLM API: {self.api_url}")
            logger.error(f"   Model: {self.model}")
            return None
        except Exception as e:
            logger.error(f"❌ Error calling LLM: {type(e).__name__}: {e}")
            logger.error(f"   Model: {self.model}")
            logger.error(f"   API: {self.api_url}")
            return None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using LLM-based relevance scoring
        
        Documents are scored using a comprehensive RAG evaluation framework (0-100 scale).
        Only documents scoring >= 50/100 are included in the results.
        
        Args:
            query: Original query
            documents: List of documents from retriever (with 'text', 'score', etc.)
            top_k: Number of top documents to return (uses self.top_k if not provided)
        
        Returns:
            Reranked documents sorted by LLM score (only docs with score >= 50/100)
            Returns empty list if no documents score >= 50
        """
        if top_k is None:
            top_k = self.top_k
        
        if not documents:
            logger.warning("⚠️ No documents to rerank")
            return []
        
        try:
            logger.info(f"🔄 Reranking {len(documents)} documents using {self.model}...")
            
            # Create rerank prompt
            prompt = self._create_rerank_prompt(query, documents)
            
            # Call LLM
            logger.debug(f"📝 Sending prompt to LLM...")
            llm_response = self._call_llm(prompt)
            
            if not llm_response:
                logger.error("❌ No response from LLM, returning original order")
                return documents[:top_k]
            
            logger.debug(f"✅ Received LLM response")
            
            # Parse scores (pass number of documents)
            scores = self._parse_scores(llm_response, len(documents))
            
            if not scores or len(scores) != len(documents):
                logger.warning(f"⚠️ Invalid score count: expected {len(documents)}, got {len(scores) if scores else 0}")
                return documents[:top_k]
            
            # Add LLM scores to documents
            reranked_docs = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                doc_copy = doc.copy()
                doc_copy['llm_score'] = score
                doc_copy['original_rank'] = i + 1
                reranked_docs.append(doc_copy)
            
            # Sort by LLM score (descending)
            reranked_docs.sort(key=lambda x: x['llm_score'], reverse=True)
            
            # Filter out documents with score below 50/100 (quality threshold)
            filtered_docs = [doc for doc in reranked_docs if doc['llm_score'] >= 50.0]
            
            if len(filtered_docs) < len(reranked_docs):
                excluded_count = len(reranked_docs) - len(filtered_docs)
                logger.info(f"🔍 Filtered out {excluded_count} document(s) with score < 50/100")
            
            # Take top_k from filtered documents
            final_docs = filtered_docs[:top_k]
            
            # Update ranks for final documents
            for new_rank, doc in enumerate(final_docs, 1):
                doc['reranked_rank'] = new_rank
            
            if not final_docs:
                logger.warning(f"⚠️ No documents scored >= 50/100. Returning empty result.")
                return []
            
            logger.info(f"✅ Reranking complete! Top {len(final_docs)} documents (score >= 50):")
            for doc in final_docs:
                logger.info(f"   Rank {doc['reranked_rank']}: LLM Score {doc['llm_score']:.1f} "
                           f"(Original: {doc['original_rank']}, Retriever Score: {doc.get('score', 'N/A')})")
            
            return final_docs
        
        except Exception as e:
            logger.error(f"❌ Error during reranking: {e}")
            return documents[:top_k]
    
    def rerank_batch(self, 
                    queries: List[str], 
                    document_batches: List[List[Dict[str, Any]]],
                    top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple batches of documents
        
        Args:
            queries: List of queries
            document_batches: List of document lists to rerank
            top_k: Number of top documents per batch
        
        Returns:
            List of reranked document lists
        """
        logger.info(f"🔄 Reranking {len(queries)} query batches...")
        
        results = []
        for i, (query, docs) in enumerate(zip(queries, document_batches), 1):
            logger.debug(f"  [{i}/{len(queries)}] Reranking {len(docs)} documents...")
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get reranker configuration"""
        return {
            'model': self.model,
            'api_url': self.api_url,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'timeout': self.timeout
        }


if __name__ == "__main__":
    # Test reranker
    reranker = LLMReranker(
        model="qwen2.5-coder:3b",
        top_k=5
    )
    
    logger.info("🧪 Testing LLM Reranker")
    logger.info("=" * 50)
