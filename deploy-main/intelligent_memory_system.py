#!/usr/bin/env python3
"""
Intelligent Memory System for TradeAI Companion
A sophisticated memory system that functions like ChatGPT's memory capabilities
Provides contextual, semantic, and long-term memory for enhanced user interactions

Features:
- Semantic memory with vector embeddings
- Contextual memory with intelligent retrieval
- User preference learning and adaptation
- Cross-session memory persistence
- Privacy-aware memory management
- Memory consolidation and optimization
"""

import json
import asyncio
import hashlib
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available in intelligent_memory_system")
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Boolean, Float, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.future import select

from db import Base, AsyncSessionLocal
from models import User
from logger import BotLogger
from performance_cache import cache_result, with_connection_pool

logger = BotLogger(__name__)

class MemoryType(Enum):
    """Types of memory entries"""
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    CONTEXT = "context"
    ALERT = "alert"
    TRADING = "trading"
    QUERY = "query"
    ERROR = "error"
    INSIGHT = "insight"
    GOAL = "goal"
    PATTERN = "pattern"

class MemoryImportance(Enum):
    """Importance levels for memory entries"""
    CRITICAL = "critical"  # Essential information, never forget
    HIGH = "high"         # Important information, long retention
    MEDIUM = "medium"     # Regular information, medium retention
    LOW = "low"           # Background information, short retention

@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    user_id: int
    memory_type: MemoryType
    content: str
    context: Optional[str] = None
    importance: MemoryImportance = MemoryImportance.MEDIUM
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

class UserMemory(Base):
    """SQLAlchemy model for user memories"""
    __tablename__ = 'user_memories'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    memory_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    context = Column(Text)
    embedding = Column(Text)  # JSON serialized embedding
    importance = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    relevance_score = Column(Float, default=1.0)
    tags = Column(Text)  # JSON serialized list
    memory_metadata = Column(Text)  # JSON serialized dict

class MemoryInsight(Base):
    """SQLAlchemy model for learned insights about users"""
    __tablename__ = 'memory_insights'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    insight_type = Column(String(50), nullable=False)  # 'preference', 'pattern', 'behavior'
    insight_data = Column(Text, nullable=False)  # JSON serialized insight
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class IntelligentMemorySystem:
    """Core intelligent memory system with semantic search and learning capabilities"""
    
    def __init__(self):
        self.embedding_cache = {}  # Cache for embeddings
        self.similarity_threshold = 0.7  # Threshold for memory consolidation
        self.max_memories_per_user = 10000  # Maximum memories per user
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using simple embeddings")
            self.embedding_model = None
    
    async def _generate_embedding(self, text: str) -> Optional[Any]:
        """Generate embedding for text"""
        if not text:
            return list(0 for _ in range(384))  # Default embedding size
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            if self.embedding_model:
                embedding = self.embedding_model.encode(text)
            else:
                # Fallback: simple hash-based embedding
                embedding = np.array([hash(text[i:i+4]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 4)])
                if len(embedding) < 384:
                    embedding = np.pad(embedding, (0, 384 - len(embedding)))
            
            # Cache the embedding
            self.embedding_cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return list(0 for _ in range(384))
    
    def _cosine_similarity(self, a: Optional[Any], b: Optional[Any]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    async def add_memory(self, memory_entry: MemoryEntry) -> bool:
        """Add a new memory entry"""
        try:
            async with AsyncSessionLocal() as session:
                # Generate embedding
                embedding = await self._generate_embedding(memory_entry.content)
                
                # Create database entry
                db_memory = UserMemory(
                     user_id=memory_entry.user_id,
                     memory_type=memory_entry.memory_type.value,
                     content=memory_entry.content,
                     context=memory_entry.context,
                     embedding=json.dumps(embedding.tolist()),
                     importance=memory_entry.importance.value,
                     tags=json.dumps(memory_entry.tags),
                     memory_metadata=json.dumps(memory_entry.metadata)
                 )
                
                session.add(db_memory)
                await session.commit()
                
                logger.info(f"Memory added: user={memory_entry.user_id}, type={memory_entry.memory_type.value}")
                
                # Check if we need to consolidate memories
                await self._check_memory_limits(memory_entry.user_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return False
    
    async def search_memories(self, user_id: int, query: str, memory_types: Optional[List[MemoryType]] = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant memories using semantic similarity"""
        try:
            async with AsyncSessionLocal() as session:
                # Build query
                query_stmt = select(UserMemory).where(UserMemory.user_id == user_id)
                
                if memory_types:
                    type_values = [mt.value for mt in memory_types]
                    query_stmt = query_stmt.where(UserMemory.memory_type.in_(type_values))
                
                result = await session.execute(query_stmt)
                memories = result.scalars().all()
                
                if not memories:
                    return []
                
                # Generate query embedding
                query_embedding = await self._generate_embedding(query)
                
                # Calculate similarities
                scored_memories = []
                for memory in memories:
                    try:
                        memory_embedding = np.array(json.loads(memory.embedding))
                        similarity = self._cosine_similarity(query_embedding, memory_embedding)
                        
                        # Update access statistics
                        memory.last_accessed = datetime.utcnow()
                        memory.access_count += 1
                        
                        scored_memories.append({
                             'id': memory.id,
                             'content': memory.content,
                             'context': memory.context,
                             'memory_type': memory.memory_type,
                             'importance': memory.importance,
                             'similarity': similarity,
                             'created_at': memory.created_at,
                             'tags': json.loads(memory.tags) if memory.tags else [],
                             'metadata': json.loads(memory.memory_metadata) if memory.memory_metadata else {}
                         })
                    except Exception as e:
                        logger.warning(f"Error processing memory {memory.id}: {e}")
                        continue
                
                # Sort by similarity and return top results
                scored_memories.sort(key=lambda x: x['similarity'], reverse=True)
                await session.commit()
                
                return scored_memories[:limit]
                
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def get_contextual_memories(self, user_id: int, context: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories relevant to current context"""
        return await self.search_memories(user_id, context, limit=limit)
    
    async def consolidate_memories(self, user_id: int) -> int:
        """Consolidate similar memories to reduce redundancy"""
        try:
            async with AsyncSessionLocal() as session:
                # Get all memories for user
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                memories = result.scalars().all()
                
                if len(memories) < 2:
                    return 0
                
                consolidated_count = 0
                memories_to_remove = set()
                
                # Compare memories pairwise
                for i, memory1 in enumerate(memories):
                    if memory1.id in memories_to_remove:
                        continue
                        
                    for j, memory2 in enumerate(memories[i+1:], i+1):
                        if memory2.id in memories_to_remove:
                            continue
                        
                        # Check if memories are similar
                        try:
                            emb1 = np.array(json.loads(memory1.embedding))
                            emb2 = np.array(json.loads(memory2.embedding))
                            similarity = self._cosine_similarity(emb1, emb2)
                            
                            if similarity > self.similarity_threshold:
                                # Consolidate memories
                                if memory1.importance == memory2.importance:
                                    # Keep the more recent one
                                    if memory1.created_at > memory2.created_at:
                                        memories_to_remove.add(memory2.id)
                                    else:
                                        memories_to_remove.add(memory1.id)
                                        break
                                else:
                                    # Keep the more important one
                                    importance_order = ['low', 'medium', 'high', 'critical']
                                    if importance_order.index(memory1.importance) > importance_order.index(memory2.importance):
                                        memories_to_remove.add(memory2.id)
                                    else:
                                        memories_to_remove.add(memory1.id)
                                        break
                                
                                consolidated_count += 1
                        except Exception as e:
                            logger.warning(f"Error comparing memories {memory1.id} and {memory2.id}: {e}")
                            continue
                
                # Remove consolidated memories
                for memory_id in memories_to_remove:
                    await session.execute(
                        select(UserMemory).where(UserMemory.id == memory_id)
                    )
                    memory = (await session.execute(
                        select(UserMemory).where(UserMemory.id == memory_id)
                    )).scalar_one_or_none()
                    if memory:
                        await session.delete(memory)
                
                await session.commit()
                logger.info(f"Consolidated {consolidated_count} memories for user {user_id}")
                return consolidated_count
                
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return 0
    
    @with_connection_pool
    async def forget_old_memories(self, user_id: int, days_threshold: int = 90) -> int:
        """Remove old, low-importance memories"""
        try:
            async with AsyncSessionLocal() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
                
                # Delete old, low-importance memories
                result = await session.execute(
                    select(UserMemory).where(
                        UserMemory.user_id == user_id,
                        UserMemory.importance.in_(['low', 'medium']),
                        UserMemory.last_accessed < cutoff_date,
                        UserMemory.access_count < 3
                    )
                )
                old_memories = result.scalars().all()
                
                for memory in old_memories:
                    await session.delete(memory)
                
                await session.commit()
                
                forgotten_count = len(old_memories)
                logger.info(f"Forgot {forgotten_count} old memories for user {user_id}")
                return forgotten_count
                
        except Exception as e:
            logger.error(f"Error forgetting old memories: {e}")
            return 0
    
    async def _check_memory_limits(self, user_id: int):
        """Check and enforce memory limits per user"""
        try:
            async with AsyncSessionLocal() as session:
                # Count memories for user
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                memory_count = len(result.scalars().all())
                
                if memory_count > self.max_memories_per_user:
                    # Remove oldest, least important memories
                    excess = memory_count - self.max_memories_per_user
                    
                    result = await session.execute(
                        select(UserMemory)
                        .where(UserMemory.user_id == user_id)
                        .where(UserMemory.importance.in_(['low', 'medium']))
                        .order_by(UserMemory.last_accessed.asc())
                        .limit(excess)
                    )
                    memories_to_remove = result.scalars().all()
                    
                    for memory in memories_to_remove:
                        await session.delete(memory)
                    
                    await session.commit()
                    logger.info(f"Removed {len(memories_to_remove)} excess memories for user {user_id}")
                    
        except Exception as e:
            logger.error(f"Error checking memory limits: {e}")
    
    async def learn_user_patterns(self, user_id: int) -> Dict[str, Any]:
        """Learn patterns from user's memory and generate insights"""
        try:
            async with AsyncSessionLocal() as session:
                # Get user memories
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                memories = result.scalars().all()
                
                if not memories:
                    return {}
                
                patterns = {
                    'memory_types': defaultdict(int),
                    'common_topics': defaultdict(int),
                    'activity_times': defaultdict(int),
                    'preferences': [],
                    'trading_patterns': [],
                    'interaction_frequency': 0
                }
                
                # Analyze memory patterns
                for memory in memories:
                    patterns['memory_types'][memory.memory_type] += 1
                    
                    # Extract topics from content
                    words = memory.content.lower().split()
                    for word in words:
                        if len(word) > 4:  # Only consider longer words
                            patterns['common_topics'][word] += 1
                    
                    # Analyze activity times
                    hour = memory.created_at.hour
                    patterns['activity_times'][hour] += 1
                    
                    # Extract preferences
                    if memory.memory_type == 'preference':
                        patterns['preferences'].append(memory.content)
                    
                    # Extract trading patterns
                    if memory.memory_type == 'trading':
                        patterns['trading_patterns'].append(memory.content)
                
                patterns['interaction_frequency'] = len(memories)
                
                # Store insights
                insight = MemoryInsight(
                    user_id=user_id,
                    insight_type='behavioral_patterns',
                    insight_data=json.dumps(dict(patterns)),
                    confidence_score=min(len(memories) / 100.0, 1.0)  # Confidence based on data amount
                )
                
                session.add(insight)
                await session.commit()
                
                return dict(patterns)
                
        except Exception as e:
            logger.error(f"Error learning user patterns: {e}")
            return {}
    
    async def get_user_insights(self, user_id: int) -> List[Dict[str, Any]]:
        """Get learned insights about a user"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(MemoryInsight).where(MemoryInsight.user_id == user_id)
                )
                insights = result.scalars().all()
                
                return [{
                    'insight_type': insight.insight_type,
                    'insight_data': json.loads(insight.insight_data),
                    'confidence_score': insight.confidence_score,
                    'created_at': insight.created_at,
                    'last_updated': insight.last_updated
                } for insight in insights]
                
        except Exception as e:
            logger.error(f"Error getting user insights: {e}")
            return []
    
    async def update_memory_importance(self, memory_id: int, new_importance: MemoryImportance) -> bool:
        """Update the importance level of a memory"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserMemory).where(UserMemory.id == memory_id)
                )
                memory = result.scalar_one_or_none()
                
                if memory:
                    memory.importance = new_importance.value
                    await session.commit()
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            return False
    
    async def get_memory_stats(self, user_id: int) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                memories = result.scalars().all()
                
                if not memories:
                    return {'total_memories': 0}
                
                stats = {
                    'total_memories': len(memories),
                    'by_type': defaultdict(int),
                    'by_importance': defaultdict(int),
                    'oldest_memory': min(m.created_at for m in memories),
                    'newest_memory': max(m.created_at for m in memories),
                    'total_access_count': sum(m.access_count for m in memories),
                    'average_relevance': sum(m.relevance_score for m in memories) / len(memories)
                }
                
                for memory in memories:
                    stats['by_type'][memory.memory_type] += 1
                    stats['by_importance'][memory.importance] += 1
                
                return dict(stats)
                
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'total_memories': 0}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the memory system"""
        try:
            # Test embedding generation
            test_embedding = await self._generate_embedding("test")
            embedding_ok = len(test_embedding) > 0
            
            # Test database connection
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(UserMemory).limit(1))
                db_ok = True
            
            return {
                'status': 'healthy' if embedding_ok and db_ok else 'unhealthy',
                'embedding_model': self.embedding_model is not None,
                'database_connection': db_ok,
                'cache_size': len(self.embedding_cache)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

# Global instance
intelligent_memory = IntelligentMemorySystem()