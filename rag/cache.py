"""
Cache Manager for RAG Pipeline
================================

Cơ chế caching để lưu trữ câu hỏi và câu trả lời nhằm:
1. Giảm latency cho các câu hỏi lặp lại
2. Tiết kiệm chi phí API calls
3. Cải thiện trải nghiệm người dùng

Hỗ trợ:
- SQLite backend (persistent)
- In-memory cache (fast)
- TTL (Time-To-Live) cho cache entries
- Cache statistics (hit/miss tracking)
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Quản lý cache cho RAG pipeline với SQLite backend
    
    Features:
    - Hash-based key generation
    - TTL support
    - Cache statistics
    - Thread-safe operations
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 db_name: str = "rag_cache.db",
                 ttl_hours: int = 24,
                 enable_cache: bool = True):
        """
        Initialize Cache Manager
        
        Args:
            cache_dir: Directory to store cache database
            db_name: SQLite database filename
            ttl_hours: Time-to-live in hours (default: 24)
            enable_cache: Enable/disable caching
        """
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / db_name
        self.ttl_hours = ttl_hours
        self.enable_cache = enable_cache
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0
        }
        
        # Initialize database
        if self.enable_cache:
            self._init_database()
            logger.info(f"✅ Cache initialized: {self.db_path}")
            logger.info(f"   TTL: {ttl_hours} hours")
        else:
            logger.info("⚠️  Cache disabled")
    
    def _init_database(self):
        """Create cache directory and database tables"""
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hits INTEGER DEFAULT 0,
                misses INTEGER DEFAULT 0,
                total_queries INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Initialize stats if not exists
        cursor.execute("SELECT COUNT(*) FROM cache_stats")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO cache_stats (hits, misses, total_queries)
                VALUES (0, 0, 0)
            """)
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(self, query: str, prefix: str = "") -> str:
        """
        Generate hash-based cache key from query
        
        Args:
            query: User query
            prefix: Optional prefix for key
        
        Returns:
            SHA-256 hash of query
        """
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        
        # Generate hash
        hash_object = hashlib.sha256(normalized_query.encode('utf-8'))
        cache_key = hash_object.hexdigest()
        
        if prefix:
            cache_key = f"{prefix}_{cache_key}"
        
        return cache_key
    
    def _is_expired(self, created_at: str) -> bool:
        """
        Check if cache entry is expired based on TTL
        
        Args:
            created_at: Timestamp string from database
        
        Returns:
            True if expired, False otherwise
        """
        created_time = datetime.fromisoformat(created_at)
        expiry_time = created_time + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached answer for query
        
        Args:
            query: User query
        
        Returns:
            Cached result dict or None if not found/expired
        """
        if not self.enable_cache:
            return None
        
        self.stats['total_queries'] += 1
        
        try:
            cache_key = self._generate_cache_key(query)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, answer, context, metadata, created_at, access_count
                FROM cache
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            
            if row:
                query_text, answer, context, metadata_json, created_at, access_count = row
                
                # Check if expired
                if self._is_expired(created_at):
                    logger.info(f"🗑️  Cache expired for query: {query[:50]}...")
                    self._delete(cache_key)
                    conn.close()
                    self.stats['misses'] += 1
                    return None
                
                # Update access stats
                cursor.execute("""
                    UPDATE cache
                    SET accessed_at = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                # Parse metadata
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                self.stats['hits'] += 1
                logger.info(f"✅ Cache HIT for query: {query[:50]}... (accessed {access_count + 1} times)")
                
                conn.close()
                
                return {
                    'query': query_text,
                    'answer': answer,
                    'context': context,
                    'metadata': metadata,
                    'from_cache': True
                }
            else:
                self.stats['misses'] += 1
                logger.info(f"❌ Cache MISS for query: {query[:50]}...")
                conn.close()
                return None
        
        except Exception as e:
            logger.error(f"❌ Error reading from cache: {e}")
            return None
    
    def set(self, 
            query: str, 
            answer: str, 
            context: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None):
        """
        Store query-answer pair in cache
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context (optional)
            metadata: Additional metadata (optional)
        """
        if not self.enable_cache:
            return
        
        try:
            cache_key = self._generate_cache_key(query)
            metadata_json = json.dumps(metadata) if metadata else None
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache 
                (cache_key, query, answer, context, metadata, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
            """, (cache_key, query, answer, context, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Cached answer for query: {query[:50]}...")
        
        except Exception as e:
            logger.error(f"❌ Error writing to cache: {e}")
    
    def _delete(self, cache_key: str):
        """Delete a cache entry by key"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Error deleting cache entry: {e}")
    
    def clear(self):
        """Clear all cache entries"""
        if not self.enable_cache:
            return
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache")
            conn.commit()
            conn.close()
            logger.info("🗑️  All cache cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        if not self.enable_cache:
            return 0
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Calculate expiry timestamp
            expiry_time = datetime.now() - timedelta(hours=self.ttl_hours)
            
            cursor.execute("""
                DELETE FROM cache
                WHERE created_at < ?
            """, (expiry_time.isoformat(),))
            
            removed_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if removed_count > 0:
                logger.info(f"🗑️  Cleaned up {removed_count} expired cache entries")
            
            return removed_count
        
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with hit/miss rates and cache size
        """
        stats = self.stats.copy()
        
        if self.enable_cache:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Get cache size
                cursor.execute("SELECT COUNT(*) FROM cache")
                stats['cache_size'] = cursor.fetchone()[0]
                
                # Get total access count
                cursor.execute("SELECT SUM(access_count) FROM cache")
                result = cursor.fetchone()[0]
                stats['total_accesses'] = result if result else 0
                
                conn.close()
            except Exception as e:
                logger.error(f"❌ Error getting cache stats: {e}")
                stats['cache_size'] = 0
                stats['total_accesses'] = 0
        else:
            stats['cache_size'] = 0
            stats['total_accesses'] = 0
        
        # Calculate hit rate
        if stats['total_queries'] > 0:
            stats['hit_rate'] = (stats['hits'] / stats['total_queries']) * 100
        else:
            stats['hit_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print cache statistics in a formatted way"""
        stats = self.get_stats()
        
        logger.info("\n" + "="*50)
        logger.info("📊 CACHE STATISTICS")
        logger.info("="*50)
        logger.info(f"Cache Enabled: {'Yes' if self.enable_cache else 'No'}")
        logger.info(f"Cache Size: {stats['cache_size']} entries")
        logger.info(f"TTL: {self.ttl_hours} hours")
        logger.info(f"\nQuery Stats:")
        logger.info(f"  Total Queries: {stats['total_queries']}")
        logger.info(f"  Cache Hits: {stats['hits']}")
        logger.info(f"  Cache Misses: {stats['misses']}")
        logger.info(f"  Hit Rate: {stats['hit_rate']:.2f}%")
        logger.info(f"  Total Accesses: {stats['total_accesses']}")
        logger.info("="*50 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize cache
    cache = CacheManager(
        cache_dir="cache",
        ttl_hours=24,
        enable_cache=True
    )
    
    # Test cache operations
    query1 = "What is BGP protocol?"
    
    # First access - Cache MISS
    result = cache.get(query1)
    print(f"First access: {result}")
    
    # Store in cache
    cache.set(
        query=query1,
        answer="BGP (Border Gateway Protocol) is a routing protocol...",
        context="BGP is used for routing between autonomous systems...",
        metadata={'model': 'qwen2.5-coder', 'tokens': 150}
    )
    
    # Second access - Cache HIT
    result = cache.get(query1)
    print(f"\nSecond access:")
    print(f"Answer: {result['answer']}")
    print(f"From cache: {result['from_cache']}")
    
    # Print statistics
    cache.print_stats()
