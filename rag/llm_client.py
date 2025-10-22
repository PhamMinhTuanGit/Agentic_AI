"""
LLM Client for RAG Pipeline
============================

Client để gọi LLM API (Ollama/OpenAI) với các tính năng:
1. Timeout handling
2. Retry mechanism
3. Streaming support
4. Error handling
5. Logging và monitoring
"""

import os
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List, Generator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CLI output configuration
from rag.cli_output_config import CLIOutputConfig, create_cli_prompt


class LLMClient:
    """
    Client để gọi LLM API với error handling và retry logic
    
    Hỗ trợ:
    - Ollama API
    - OpenAI API (extensible)
    - Streaming responses
    - Automatic retries
    """
    
    def __init__(
                 self,
                 api_url: str = "http://localhost:11434/api/generate",
                 model: str = "qwen2.5-coder:3b",
                 temperature: float = 0.7,
                 timeout: int = 60,
                 max_retries: int = 3,
                 max_tokens: int = 2048,
                 retry_delay: float = 1.0
             ):
        """
        Initialize LLM Client
        
        Args:
            api_url: LLM API endpoint
            model: Model name
            temperature: Sampling temperature (0-1)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            max_tokens: Maximum tokens in response
            retry_delay: Delay between retries (seconds)
        """
        self.api_url = api_url or os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        
        logger.info(f"✅ LLM Client initialized")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   API: {self.api_url}")
        logger.info(f"   Temperature: {self.temperature}")
        logger.info(f"   Max Tokens: {self.max_tokens}")
    
    def _build_prompt(self, 
                     query: str, 
                     context: str,
                     system_prompt: Optional[str] = None) -> str:
        """
        Build prompt from query and context
        
        Args:
            query: User question
            context: Retrieved context
            system_prompt: Optional system instructions
        
        Returns:
            Formatted prompt
        """
        if system_prompt is None:
            system_prompt = """You are an expert ZebOS network device configuration assistant specialized in:

## Expertise Areas:
1. **ZebOS CLI Configuration**: Generating CLI commands for network device configuration
2. **Routing Protocols**: BGP, OSPF, EIGRP, IS-IS, RIP configuration
3. **Network Interfaces**: Port configuration, VLAN setup, LAG/Port-channel
4. **ACLs & Security**: Access control lists, firewall rules, AAA authentication
5. **QoS**: Quality of Service policies and traffic shaping
6. **High Availability**: Redundancy, failover, and clustering
7. **Monitoring**: SNMP, syslog, NetFlow configuration

## Response Format:
- **For CLI requests**: Provide exact, executable ZebOS commands with clear syntax
- **For configuration**: Include step-by-step instructions with proper command ordering
- **For troubleshooting**: Suggest diagnostic commands and validation steps
- **For examples**: Show complete configuration blocks when relevant

## ZebOS Command Syntax:
- Use "configure" (not "configure terminal")
- Use "ipv4 address" (not "ip address")
- Use "interface ethernet" (not just "interface")
- Use "exit" to exit configuration modes
- Device prompts: R1#, R1(config)#, R1(config-if)#, etc.

## Instructions:
1. Answer based ONLY on the provided context
2. When generating CLI commands, ensure they are valid ZebOS syntax
3. Include comments (!) to explain complex configurations
4. If the answer is not in the context, say "I don't have enough information to answer this question"
5. Be concise, accurate, and provide working configurations
6. For multi-step configurations, number the steps clearly
7. Highlight important warnings or prerequisites with ⚠️
8. Cite relevant parts of the context when appropriate

## Output Format for CLI Commands:
- Use code blocks for commands
- Prefix with device type/context (e.g., Router#, Switch#, Interface#)
- Give complete commands, not just fragments
- Separate commands with new lines for readability
- Include output validation where applicable

**Example**:
```zsh
! Step 1: Enter Configure mode and setup OSPF
R1#configure
R1(config)#router ospf 100
R1(config-router)#network 10.1.1.0 0.0.0.255 area 1
R1(config-router)#network 1.1.1.1 0.0.0.0 area 1
R1(config-router)#bfd all-interfaces
R1(config-router)#exit

! Step 2: Configure Interface
R1(config)#interface ethernet G0/0
R1(config-if)#ipv4 address 10.1.1.1 255.255.255.0
R1(config-if)#no shutdown
R1(config-if)#exit

! Step 3: Verify Configuration
R1#show ip ospf neighbor
R1#show interface brief
```
"""
        
        # Add reasoning prompts to encourage step-by-step thinking
        reasoning_prompt = """
## Reasoning Instructions:
**Hãy suy nghĩ từng bước một.** (Think step by step.)

When answering:
1. **Phân tích câu hỏi** (Analyze the question) - Break down what's being asked
2. **Xác định thông tin cần thiết** (Identify necessary information) - What information from context is relevant?
3. **Suy luận từng bước** (Reason step by step) - How does the information connect?
4. **Giải thích lý do** (Explain your reasoning) - Why is this the answer?
5. **Xây dựng câu trả lời** (Construct the answer) - Provide the complete, verified answer

**Show your thinking in your response.**
"""
        
        prompt = f"""{system_prompt}{reasoning_prompt}

Context:
{context}

Question: {query}

**Hãy suy nghĩ từng bước một. (Let me think step by step.)**

Answer:"""
        
        return prompt
    
    def generate(self,
                query: str,
                context: str,
                system_prompt: Optional[str] = None,
                stream: bool = False,
                output_format: str = "default",
                session_type: str = "general",
                use_cot: bool = False,
                cot_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate answer using LLM
        
        Args:
            query: User question
            context: Retrieved context
            system_prompt: Optional system instructions
            stream: Enable streaming response
            output_format: Output format (default, single_code_block)
            session_type: Type of session (general, router, switch, topology)
            use_cot: Enable Chain-of-Thought reasoning
            cot_prompt: Pre-generated CoT prompt (if use_cot=True)
        
        Returns:
            Dict with answer and metadata
        """
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Use Chain-of-Thought prompt if provided
            if use_cot and cot_prompt:
                prompt = cot_prompt
                logger.info("🧠 Using Chain-of-Thought reasoning")
            # Use CLI output config for single_code_block format
            elif output_format == "single_code_block":
                prompt = create_cli_prompt(query, context, session_type, output_type='single_code_block')
            elif output_format == "multi_code_block":
                prompt = create_cli_prompt(query, context, session_type, output_type='multi_code_block')

            else:
                # Build prompt using default format
                prompt = self._build_prompt(query, context, system_prompt)
            
            logger.info(f"🤖 Generating answer with {self.model}...")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Try with retries
            for attempt in range(self.max_retries):
                try:
                    response = self._call_api(prompt, stream=stream)
                    
                    if response:
                        elapsed_time = time.time() - start_time
                        
                        self.stats['successful_requests'] += 1
                        self.stats['total_time'] += elapsed_time
                        
                        result = {
                            'answer': response.get('response', ''),
                            'model': self.model,
                            'prompt_tokens': response.get('prompt_eval_count', 0),
                            'completion_tokens': response.get('eval_count', 0),
                            'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                            'elapsed_time': elapsed_time,
                            'from_cache': False
                        }
                        
                        self.stats['total_tokens'] += result['total_tokens']
                        
                        logger.info(f"✅ Answer generated in {elapsed_time:.2f}s")
                        logger.info(f"   Tokens: {result['total_tokens']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
                        
                        return result
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"⚠️  Timeout on attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise
                
                except requests.exceptions.ConnectionError:
                    logger.warning(f"⚠️  Connection error on attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise
            
            # All retries failed
            raise Exception("All retry attempts failed")
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"❌ Error generating answer: {e}")
            
            return {
                'answer': f"Error: Unable to generate answer - {str(e)}",
                'model': self.model,
                'error': str(e),
                'from_cache': False
            }
    
    def _call_api(self, prompt: str, stream: bool = False) -> Optional[Dict[str, Any]]:
        """
        Call Ollama API
        
        Args:
            prompt: Formatted prompt
            stream: Enable streaming
        
        Returns:
            API response or None
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": stream
        }
        
        # Add max_tokens if supported
        if self.max_tokens:
            payload["options"] = {
                "num_predict": self.max_tokens
            }
        
        response = requests.post(
            self.api_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json()
    
    def generate_stream(self,
                       query: str,
                       context: str,
                       system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """
        Generate answer with streaming
        
        Args:
            query: User question
            context: Retrieved context
            system_prompt: Optional system instructions
        
        Yields:
            Token strings as they are generated
        """
        prompt = self._build_prompt(query, context, system_prompt)
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": True
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
        
        except Exception as e:
            logger.error(f"❌ Error in streaming: {e}")
            yield f"Error: {str(e)}"
    
    def batch_generate(self,
                      queries: List[str],
                      contexts: List[str],
                      system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries
        
        Args:
            queries: List of questions
            contexts: List of contexts
            system_prompt: Optional system instructions
        
        Returns:
            List of answer dicts
        """
        if len(queries) != len(contexts):
            raise ValueError("Number of queries and contexts must match")
        
        logger.info(f"🔄 Generating {len(queries)} answers in batch...")
        
        results = []
        for i, (query, context) in enumerate(zip(queries, contexts), 1):
            logger.info(f"  [{i}/{len(queries)}] Processing...")
            result = self.generate(query, context, system_prompt)
            results.append(result)
        
        logger.info(f"✅ Batch generation complete")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = self.stats.copy()
        
        if stats['successful_requests'] > 0:
            stats['avg_time_per_request'] = stats['total_time'] / stats['successful_requests']
            stats['avg_tokens_per_request'] = stats['total_tokens'] / stats['successful_requests']
        else:
            stats['avg_time_per_request'] = 0.0
            stats['avg_tokens_per_request'] = 0
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = (stats['successful_requests'] / stats['total_requests']) * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print client statistics"""
        stats = self.get_stats()
        
        logger.info("\n" + "="*50)
        logger.info("📊 LLM CLIENT STATISTICS")
        logger.info("="*50)
        logger.info(f"Model: {self.model}")
        logger.info(f"\nRequests:")
        logger.info(f"  Total: {stats['total_requests']}")
        logger.info(f"  Successful: {stats['successful_requests']}")
        logger.info(f"  Failed: {stats['failed_requests']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.2f}%")
        logger.info(f"\nPerformance:")
        logger.info(f"  Total Time: {stats['total_time']:.2f}s")
        logger.info(f"  Avg Time/Request: {stats['avg_time_per_request']:.2f}s")
        logger.info(f"  Total Tokens: {stats['total_tokens']}")
        logger.info(f"  Avg Tokens/Request: {stats['avg_tokens_per_request']:.0f}")
        logger.info("="*50 + "\n")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0
        }
        logger.info("🔄 Statistics reset")


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = LLMClient(
        model="qwen2.5-coder:3b",
        temperature=0.7,
        timeout=60
    )
    
    # Test generation
    query = "What is BGP?"
    context = "BGP (Border Gateway Protocol) is a routing protocol used for exchanging routing information between autonomous systems on the Internet."
    
    result = client.generate(query, context)
    
    print(f"\nQuery: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result.get('total_tokens', 'N/A')}")
    
    # Print stats
    client.print_stats()
