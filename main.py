"""
RAG Pipeline Entry Point
=========================

Main application để chạy RAG pipeline với interface CLI hoặc API

Usage:
    # CLI mode
    python main.py --query "What is BGP?"
    
    # Interactive mode
    python main.py --interactive
    
    # API server mode
    python main.py --api --port 8000
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGApplication:
    """
    RAG Application với multiple interfaces
    """
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize application
        
        Args:
            pipeline: RAG Pipeline instance
        """
        self.pipeline = pipeline
    
    def cli_mode(self, query: str, verbose: bool = False):
        """
        CLI mode: Single query
        
        Args:
            query: User question
            verbose: Show detailed information
        """
        logger.info("\n" + "="*70)
        logger.info("🤖 RAG PIPELINE - CLI MODE")
        logger.info("="*70)
        
        result = self.pipeline.query(
            question=query,
            return_context=verbose,
            return_sources=verbose
        )
        
        # Display result
        print("\n" + "="*70)
        print("📝 QUESTION")
        print("="*70)
        print(result['question'])
        
        print("\n" + "="*70)
        print("💬 ANSWER")
        print("="*70)
        print(result['answer'])
        
        print("\n" + "="*70)
        print("📊 METADATA")
        print("="*70)
        print(f"From Cache: {result['from_cache']}")
        print(f"Elapsed Time: {result['elapsed_time']:.2f}s")
        
        if 'model' in result:
            print(f"Model: {result['model']}")
            print(f"Tokens: {result.get('tokens', 'N/A')}")
        
        if 'breakdown' in result:
            print("\nTime Breakdown:")
            for step, time_val in result['breakdown'].items():
                print(f"  {step.capitalize()}: {time_val:.2f}s")
        
        if verbose and 'sources' in result:
            print("\n" + "="*70)
            print("📚 SOURCES")
            print("="*70)
            for i, source in enumerate(result['sources'], 1):
                print(f"\n[Source {i}]")
                print(f"LLM Score: {source.get('llm_score', 'N/A')}")
                print(f"Retriever Score: {source.get('retriever_score', 'N/A'):.4f}")
                print(f"Text: {source['text']}")
        
        print("\n" + "="*70 + "\n")
    
    def interactive_mode(self):
        """
        Interactive mode: Multiple queries
        """
        logger.info("\n" + "="*70)
        logger.info("🤖 RAG PIPELINE - INTERACTIVE MODE")
        logger.info("="*70)
        logger.info("\nCommands:")
        logger.info("  - Type your question and press Enter")
        logger.info("  - Type 'stats' to see pipeline statistics")
        logger.info("  - Type 'cache' to see cache statistics")
        logger.info("  - Type 'clear' to clear cache")
        logger.info("  - Type 'quit' or 'exit' to quit")
        logger.info("="*70 + "\n")
        
        while True:
            try:
                # Get user input
                query = input("\n💬 You: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("\n👋 Goodbye!")
                    break
                
                elif query.lower() == 'stats':
                    self.pipeline.print_stats()
                    continue
                
                elif query.lower() == 'cache':
                    self.pipeline.cache.print_stats()
                    continue
                
                elif query.lower() == 'clear':
                    self.pipeline.cache.clear()
                    logger.info("✅ Cache cleared")
                    continue
                
                elif query.lower() == 'help':
                    logger.info("\nAvailable commands:")
                    logger.info("  stats  - Show pipeline statistics")
                    logger.info("  cache  - Show cache statistics")
                    logger.info("  clear  - Clear cache")
                    logger.info("  quit   - Exit application")
                    continue
                
                # Process query
                result = self.pipeline.query(query)
                
                # Display answer
                print(f"\n🤖 Assistant: {result['answer']}")
                
                cache_indicator = "💾" if result['from_cache'] else "🔄"
                print(f"\n{cache_indicator} [{result['elapsed_time']:.2f}s]", end="")
                if result['from_cache']:
                    print(" (from cache)", end="")
                print()
            
            except KeyboardInterrupt:
                logger.info("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"\n❌ Error: {e}")
                continue
        
        # Show final statistics
        print("\n" + "="*70)
        print("📊 SESSION STATISTICS")
        print("="*70)
        self.pipeline.print_stats()
    
    def batch_mode(self, questions_file: str, output_file: str = None):
        """
        Batch mode: Process questions from file
        
        Args:
            questions_file: Path to file with questions (one per line)
            output_file: Path to output JSON file
        """
        logger.info("\n" + "="*70)
        logger.info("🤖 RAG PIPELINE - BATCH MODE")
        logger.info("="*70)
        
        # Read questions
        with open(questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"\n📝 Processing {len(questions)} questions...")
        
        # Process batch
        results = self.pipeline.batch_query(questions)
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        logger.info("\n📊 BATCH SUMMARY")
        logger.info("="*70)
        
        cache_hits = sum(1 for r in results if r['from_cache'])
        total_time = sum(r['elapsed_time'] for r in results)
        
        logger.info(f"Total Questions: {len(questions)}")
        logger.info(f"Cache Hits: {cache_hits}")
        logger.info(f"Cache Misses: {len(questions) - cache_hits}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Avg Time/Question: {total_time/len(questions):.2f}s")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline with Caching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode (default)'
    )
    mode_group.add_argument(
        '--query', '-q',
        type=str,
        help='Single query mode'
    )
    mode_group.add_argument(
        '--batch', '-b',
        type=str,
        metavar='FILE',
        help='Batch mode (process questions from file)'
    )
    
    # Options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for batch mode'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (show sources and context)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    # Pipeline configuration
    parser.add_argument(
        '--retriever-top-k',
        type=int,
        default=10,
        help='Number of documents to retrieve (default: 10)'
    )
    parser.add_argument(
        '--reranker-top-k',
        type=int,
        default=5,
        help='Number of documents after reranking (default: 5)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen2.5-coder:3b',
        help='LLM model name (default: qwen2.5-coder:3b)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        logger.info("🚀 Starting RAG Application...")
        
        pipeline = RAGPipeline(
            retriever_top_k=args.retriever_top_k,
            reranker_top_k=args.reranker_top_k,
            llm_model=args.model,
            llm_temperature=args.temperature,
            enable_cache=not args.no_cache
        )
        
        app = RAGApplication(pipeline)
        
        # Run appropriate mode
        if args.query:
            app.cli_mode(args.query, verbose=args.verbose)
        
        elif args.batch:
            app.batch_mode(args.batch, output_file=args.output)
        
        else:
            # Default to interactive mode
            app.interactive_mode()
    
    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
