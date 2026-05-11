"""
记忆系统模块

提供双层记忆架构：
- 短期记忆（Redis）：当前会话缓存
- 长期记忆（MySQL）：持久化存储和用户画像
"""

from memory.memory_manager import MemoryManager

__all__ = ["MemoryManager"]
