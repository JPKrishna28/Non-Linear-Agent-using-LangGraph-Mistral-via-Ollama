from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime
import logging
from logger_config import setup_logging

logger = setup_logging()

class Memory:
    """Memory system for storing conversation history and context"""
    def __init__(self):
        self.conversations: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.session_context: Dict[str, Any] = {}
        self.memory_file = Path("memory.json")
        self.load_memory()

    def save_memory(self):
        """Save memory to file"""
        try:
            memory_data = {
                "conversations": self.conversations[-50:],  # Keep last 50 conversations
                "user_preferences": self.user_preferences,
                "session_context": self.session_context
            }
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)
            logger.info("Memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def load_memory(self):
        """Load memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                    self.conversations = memory_data.get("conversations", [])
                    self.user_preferences = memory_data.get("user_preferences", {})
                    self.session_context = memory_data.get("session_context", {})
                logger.info("Memory loaded successfully")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")

    def add_conversation(self, query: str, response: str, route: str, metadata: Dict[str, Any] = None):
        """Add conversation to memory"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "route": route,
            "metadata": metadata or {}
        }
        self.conversations.append(conversation)
        self.save_memory()

    def get_context(self, query: str) -> str:
        """Get relevant context from memory"""
        relevant_context = []
        query_lower = query.lower()

        # Look for similar queries in recent conversations
        for conv in self.conversations[-10:]:  # Check last 10 conversations
            if any(word in conv['query'].lower() for word in query_lower.split()):
                relevant_context.append(f"Previous: {conv['query']} -> {conv['response'][:100]}...")

        return "\n".join(relevant_context) if relevant_context else ""

    def clear_memory(self):
        """Clear all memory"""
        self.conversations = []
        self.user_preferences = {}
        self.session_context = {}
        self.save_memory()
        logger.info("Memory cleared successfully")

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_conversations": len(self.conversations),
            "route_distribution": {},
            "average_response_length": 0,
            "last_interaction": None
        }

        if self.conversations:
            # Calculate route distribution
            routes = [conv["route"] for conv in self.conversations]
            for route in set(routes):
                stats["route_distribution"][route] = routes.count(route)

            # Calculate average response length
            total_length = sum(len(conv["response"]) for conv in self.conversations)
            stats["average_response_length"] = total_length / len(self.conversations)

            # Get last interaction time
            stats["last_interaction"] = self.conversations[-1]["timestamp"]

        return stats