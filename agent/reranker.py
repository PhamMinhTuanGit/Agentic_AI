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
        Create a prompt for the LLM to score document relevance
        
        Args:
            query: Original query
            documents: List of documents with text and rank
        
        Returns:
            Prompt string for the reranker model
        """
        # Check if using specialized reranker model
        is_reranker_model = "reranker" in self.model.lower()
        
        if is_reranker_model:
            # Format for Qwen3-Reranker - simpler format
            doc_text = ""
            for i, doc in enumerate(documents, 1):
                doc_text += f"[{i}] {doc['text'][:150]}\n"
            
            prompt = f"""Query: {query}

Documents:
{doc_text}

Score each document's relevance to the query (0-100). Return ONLY JSON array of scores."""
        else:
            # Format for general LLMs
            doc_text = ""
            for i, doc in enumerate(documents, 1):
                doc_text += f"[DOC {i}]\n{doc['text'][:200]}...\n\n"
            
            prompt = f"""You are a document g scorer. Given a query and a list of documents, score each document's relevance to the query from 0-100.

Query: {query}

Documents:
{doc_text}

Task: Score each document from 0-100 based on how well it answers the query. Return ONLY a JSON array of scores like this:
[85, 45, 92, 30, 78, 55, 88, 40, 70, 50]

Important: Return ONLY the JSON array, nothing else."""
        
        return prompt
    
    def _parse_scores(self, response_text: str) -> Optional[List[float]]:
        """
        Parse LLM response to extract scores
        
        Args:
            response_text: LLM response
        
        Returns:
            List of scores or None if parsing fails
        """
        try:
            # Clean response
            response_text = response_text.strip()
            
            # Find JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning(f"⚠️ No JSON array found in response")
                return None
            
            json_str = response_text[start_idx:end_idx]
            
            # Evaluate as Python literal
            import ast
            scores = ast.literal_eval(json_str)
            
            # Validate scores
            if not isinstance(scores, list):
                logger.warning(f"⚠️ Scores not a list: {type(scores)}")
                return None
            
            scores = [float(s) for s in scores]
            
            # Clamp scores to 0-100
            scores = [max(0, min(100, s)) for s in scores]
            
            return scores
        
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
        
        Args:
            query: Original query
            documents: List of documents from retriever (with 'text', 'score', etc.)
            top_k: Number of top documents to return (uses self.top_k if not provided)
        
        Returns:
            Reranked documents sorted by LLM score
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
            
            # Parse scores
            scores = self._parse_scores(llm_response)
            
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
            
            # Update ranks
            for new_rank, doc in enumerate(reranked_docs[:top_k], 1):
                doc['reranked_rank'] = new_rank
            
            logger.info(f"✅ Reranking complete! Top {top_k} documents:")
            for doc in reranked_docs[:top_k]:
                logger.info(f"   Rank {doc['reranked_rank']}: LLM Score {doc['llm_score']:.1f} "
                           f"(Original: {doc['original_rank']}, Retriever Score: {doc.get('score', 'N/A')})")
            
            return reranked_docs[:top_k]
        
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
