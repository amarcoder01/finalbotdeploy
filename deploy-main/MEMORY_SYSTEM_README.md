# 🧠 Intelligent Memory System for TradeAI Companion

A state-of-the-art memory system that enables the bot to remember and learn from user interactions, similar to ChatGPT's memory capabilities.

## 🌟 Features

### Core Memory Capabilities
- **Contextual Memory**: Remembers user preferences, trading patterns, and conversation context
- **Semantic Search**: Uses embeddings to find relevant memories based on context
- **Memory Consolidation**: Automatically merges and updates related memories
- **Intelligent Forgetting**: Removes outdated or irrelevant memories
- **Cross-Session Persistence**: Maintains memory across bot restarts and user sessions

### Memory Types
- **CONVERSATION**: General chat interactions and questions
- **PREFERENCE**: User settings and personal preferences
- **TRADING**: Trading activities and investment decisions
- **ALERT**: Alert configurations and notifications
- **QUERY**: Stock queries and market research
- **ERROR**: Error contexts for better support
- **INSIGHT**: Learned patterns and behavioral insights

### Memory Importance Levels
- **CRITICAL**: Essential information (major trades, key preferences)
- **HIGH**: Important interactions (alerts, significant queries)
- **MEDIUM**: Regular interactions (conversations, minor trades)
- **LOW**: Background information (routine queries)

## 🏗️ Architecture

### Components

1. **IntelligentMemorySystem** (`intelligent_memory_system.py`)
   - Core memory engine with semantic search
   - Memory consolidation and forgetting algorithms
   - Pattern learning and insight generation

2. **EnhancedMemoryService** (`enhanced_memory_service.py`)
   - High-level memory operations
   - Integration with existing conversation memory
   - Contextual response enhancement

3. **MemoryIntegration** (`memory_integration.py`)
   - Decorators for automatic memory capture
   - Seamless integration with bot commands
   - Memory-enhanced prompt generation

### Database Schema

#### UserMemory Table
```sql
CREATE TABLE user_memories (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    context TEXT,
    embedding BLOB,
    importance VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    relevance_score FLOAT DEFAULT 1.0,
    tags TEXT,
    metadata TEXT
);
```

#### MemoryInsight Table
```sql
CREATE TABLE memory_insights (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    insight_type VARCHAR(50) NOT NULL,
    insight_data TEXT NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🚀 Installation & Setup

### 1. Run Database Migration
```bash
python memory_migration.py
```

### 2. Install Dependencies
The memory system uses the existing dependencies. Ensure you have:
- `sentence-transformers` for embeddings
- `numpy` for similarity calculations
- `sqlalchemy` for database operations

### 3. Configuration
No additional configuration needed. The system integrates automatically with existing bot components.

## 💡 Usage Examples

### Automatic Memory Capture
The system automatically captures memories through decorators:

```python
@remember_interaction(memory_type=MemoryType.CONVERSATION, importance=MemoryImportance.MEDIUM)
async def handle_message(self, update, context):
    # Your message handling code
    pass

@remember_trading_activity(importance=MemoryImportance.HIGH)
async def trade_command(self, update, context):
    # Your trading code
    pass
```

### Manual Memory Operations
```python
# Add a memory
await memory_service.add_interaction(
    user_id=12345,
    message="User prefers technical analysis",
    response="Noted your preference for technical indicators",
    memory_type=MemoryType.PREFERENCE,
    importance=MemoryImportance.HIGH
)

# Get contextual data
context_data = await memory_service.get_contextual_response_data(
    user_id=12345,
    current_message="Show me AAPL analysis"
)
```

### Enhanced Responses
The system automatically enhances bot responses with relevant context:

```python
# Before: Generic response
"Here's the AAPL analysis..."

# After: Contextual response
"Based on your preference for technical analysis and previous interest in tech stocks, here's the AAPL analysis..."
```

## 🔧 Memory Management

### Automatic Processes
- **Memory Consolidation**: Runs every 24 hours to merge similar memories
- **Relevance Updates**: Adjusts memory importance based on access patterns
- **Cleanup**: Removes low-relevance memories older than 90 days

### Manual Management
```python
# Force memory consolidation
await memory_system.consolidate_memories(user_id=12345)

# Update memory importance
await memory_system.update_memory_importance(memory_id=123, new_importance=MemoryImportance.HIGH)

# Get memory statistics
stats = await memory_system.get_memory_stats(user_id=12345)
```

## 📊 Performance Features

### Caching
- **Embedding Cache**: Caches computed embeddings for 1 hour
- **Context Cache**: Caches contextual data for 30 minutes
- **Pattern Cache**: Caches learned patterns for 2 hours

### Optimization
- **Batch Processing**: Processes multiple memories efficiently
- **Lazy Loading**: Loads embeddings only when needed
- **Connection Pooling**: Reuses database connections

## 🔒 Privacy & Security

### Data Protection
- **User Isolation**: Each user's memories are completely isolated
- **Encryption**: Sensitive data can be encrypted before storage
- **Retention Limits**: Automatic cleanup of old memories

### Compliance
- **GDPR Ready**: Easy user data deletion
- **Configurable Retention**: Adjustable memory retention periods
- **Audit Trail**: Tracks memory access and modifications

## 🎯 Use Cases

### Trading Assistant
- Remembers user's risk tolerance and investment preferences
- Tracks trading patterns and suggests improvements
- Recalls previous analysis requests for consistency

### Personalized Alerts
- Learns optimal alert timing based on user activity
- Remembers preferred notification formats
- Adapts alert frequency to user engagement

### Contextual Conversations
- Maintains conversation context across sessions
- References previous discussions naturally
- Builds long-term relationship with users

## 🔍 Monitoring & Debugging

### Logging
The system provides comprehensive logging:
```python
logger.info(f"Memory added: user={user_id}, type={memory_type}, importance={importance}")
logger.debug(f"Semantic search found {len(results)} relevant memories")
logger.warning(f"Memory consolidation took {duration}s for user {user_id}")
```

### Health Metrics
```python
# Get memory system health
health = await memory_integration.get_memory_health(user_id=12345)
print(f"Total memories: {health['total_memories']}")
print(f"Recent activity: {health['recent_activity']}")
print(f"System performance: {health['performance_score']}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Migration Fails**
   - Check database connection
   - Verify permissions
   - Ensure SQLAlchemy is updated

2. **Slow Performance**
   - Check embedding cache hit rate
   - Monitor database query performance
   - Consider memory cleanup

3. **Memory Not Persisting**
   - Verify decorator usage
   - Check database transactions
   - Review error logs

### Debug Commands
```python
# Check memory system status
await memory_system.health_check()

# Verify embeddings
embedding = await memory_system._generate_embedding("test text")
print(f"Embedding shape: {embedding.shape}")

# Test similarity search
results = await memory_system.search_memories(user_id=12345, query="trading")
print(f"Found {len(results)} memories")
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-modal Memory**: Support for images and documents
- **Federated Learning**: Learn from anonymized user patterns
- **Advanced Analytics**: Memory usage insights and recommendations
- **API Integration**: External memory system integration

### Experimental Features

- **Predictive Memory**: Anticipate user needs based on patterns
- **Collaborative Memory**: Share insights across similar users (with consent)

## 📚 API Reference

See the individual module docstrings for detailed API documentation:
- `IntelligentMemorySystem`: Core memory operations
- `EnhancedMemoryService`: High-level memory service
- `MemoryIntegration`: Integration decorators and utilities

## 🤝 Contributing

To contribute to the memory system:
1. Follow the existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Consider privacy implications
5. Test with real user scenarios

---

**Built with ❤️ for TradeAI Companion**

*This memory system brings human-like memory capabilities to your trading bot, making every interaction more personal and contextual.*