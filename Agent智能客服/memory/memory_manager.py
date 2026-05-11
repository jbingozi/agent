"""
记忆管理系统

架构说明：
本模块实现了双层记忆架构，支持短期会话缓存和长期持久化存储。

核心组件：
1. MemoryManager: 记忆管理器主类，协调 Redis 和 MySQL 的操作
2. ShortTermMemory (Redis):
   - 存储当前会话的对话历史
   - 支持快速读写，TTL 自动过期
   - Key 格式: session:{session_id}:messages
3. LongTermMemory (MySQL):
   - conversations 表: 持久化存储所有对话
   - user_profiles 表: 用户画像和偏好
   - session_metadata 表: 会话元数据

数据流：
用户提问 -> 从 Redis 加载历史 -> LLM 处理 -> 保存到 Redis -> 异步保存到 MySQL
"""

import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import redis
import pymysql
from pymysql.cursors import DictCursor
from utilss.logger_handler import logger
from utilss.config_handler import agent_conf


class MemoryManager:
    """
    记忆管理器

    负责管理短期记忆（Redis）和长期记忆（MySQL）的读写操作
    """

    def __init__(self):
        """初始化记忆管理器，建立 Redis 和 MySQL 连接"""

        memory_config = agent_conf.get("memory", {})

        # --------------------------------------------------------------------
        # 初始化 Redis 连接（短期记忆）
        # --------------------------------------------------------------------
        redis_config = memory_config.get("redis", {})
        self.redis_client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password") or None,
            decode_responses=True,
            socket_timeout=5
        )

        self.session_ttl = redis_config.get("session_ttl", 3600)
        self.max_short_term_messages = memory_config.get("strategy", {}).get("max_short_term_messages", 50)

        # --------------------------------------------------------------------
        # 初始化 MySQL 连接（长期记忆）
        # --------------------------------------------------------------------
        mysql_config = memory_config.get("mysql", {})
        self.mysql_config = mysql_config

        try:
            self._init_database()
            logger.info("[MemoryManager] 记忆系统初始化成功")
        except Exception as e:
            logger.error(f"[MemoryManager] 记忆系统初始化失败: {e}")
            raise

    def _init_database(self):
        """初始化 MySQL 数据库和表结构"""

        connection = self._get_mysql_connection()
        try:
            with connection.cursor() as cursor:
                # 创建数据库（如果不存在）
                db_name = self.mysql_config.get("database", "agent_memory")
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cursor.execute(f"USE `{db_name}`")

                # 创建对话记录表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                        session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
                        user_id VARCHAR(64) COMMENT '用户ID',
                        role VARCHAR(16) NOT NULL COMMENT '角色: user/assistant/tool',
                        content TEXT COMMENT '消息内容',
                        metadata JSON COMMENT '元数据（工具调用等）',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        INDEX idx_session_id (session_id),
                        INDEX idx_user_id (user_id),
                        INDEX idx_created_at (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话记录表'
                """)

                # 创建用户画像表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                        user_id VARCHAR(64) NOT NULL UNIQUE COMMENT '用户ID',
                        preferences JSON COMMENT '用户偏好',
                        interests JSON COMMENT '兴趣标签',
                        conversation_count INT DEFAULT 0 COMMENT '对话次数',
                        last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后活跃时间',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        INDEX idx_user_id (user_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户画像表'
                """)

                # 创建会话元数据表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_metadata (
                                                                    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
                                                                    session_id VARCHAR(64) NOT NULL UNIQUE COMMENT '会话ID',
                        user_id VARCHAR(64) COMMENT '用户ID',
                        title VARCHAR(255) COMMENT '会话标题',
                        message_count INT DEFAULT 0 COMMENT '消息数量',
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '开始时间',
                        end_time TIMESTAMP NULL COMMENT '结束时间',
                        is_active TINYINT(1) DEFAULT 1 COMMENT '是否活跃',
                        INDEX idx_session_id (session_id),
                        INDEX idx_user_id (user_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话元数据表'
                """)

                connection.commit()
                logger.info("[MemoryManager] 数据库表结构初始化完成")
        finally:
            connection.close()

    def _get_mysql_connection(self):
        """获取 MySQL 数据库连接"""

        return pymysql.connect(
            host=self.mysql_config.get("host", "localhost"),
            port=self.mysql_config.get("port", 3306),
            user=self.mysql_config.get("user", "root"),
            password=self.mysql_config.get("password", "root"),
            database=self.mysql_config.get("database", "agent_memory"),
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=False
        )

    @contextmanager
    def _mysql_cursor(self):
        """MySQL 游标上下文管理器"""

        connection = self._get_mysql_connection()
        try:
            with connection.cursor() as cursor:
                yield cursor, connection
        finally:
            connection.close()

    # ========================================================================
    # 短期记忆操作（Redis）
    # ========================================================================

    def save_message_to_redis(self, session_id: str, role: str, content: str,
                               metadata: Optional[Dict] = None):
        """
        保存消息到 Redis（短期记忆）

        Args:
            session_id: 会话ID
            role: 角色（user/assistant/tool）
            content: 消息内容
            metadata: 元数据（可选）
        """

        try:
            key = f"session:{session_id}:messages"

            message = {
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }

            # 使用 LPUSH 添加到列表头部，保持最新消息在前
            self.redis_client.lpush(key, json.dumps(message, ensure_ascii=False))

            # 限制消息数量，避免内存溢出
            self.redis_client.ltrim(key, 0, self.max_short_term_messages - 1)

            # 设置 TTL
            self.redis_client.expire(key, self.session_ttl)

            logger.debug(f"[MemoryManager] 消息已保存到 Redis: session={session_id}, role={role}")
        except Exception as e:
            logger.error(f"[MemoryManager] 保存消息到 Redis 失败: {e}")

    def get_messages_from_redis(self, session_id: str) -> List[Dict[str, Any]]:
        """
        从 Redis 获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            List[Dict]: 消息列表（按时间正序排列）
        """

        try:
            key = f"session:{session_id}:messages"

            # 获取所有消息
            messages_json = self.redis_client.lrange(key, 0, -1)

            if not messages_json:
                return []

            # 解析 JSON 并反转顺序（因为 LPUSH 是倒序存储）
            messages = [json.loads(msg) for msg in reversed(messages_json)]

            logger.debug(f"[MemoryManager] 从 Redis 获取 {len(messages)} 条消息: session={session_id}")
            return messages
        except Exception as e:
            logger.error(f"[MemoryManager] 从 Redis 获取消息失败: {e}")
            return []

    def clear_session_redis(self, session_id: str):
        """清除 Redis 中的会话数据"""

        try:
            key = f"session:{session_id}:messages"
            self.redis_client.delete(key)
            logger.info(f"[MemoryManager] Redis 会话已清除: session={session_id}")
        except Exception as e:
            logger.error(f"[MemoryManager] 清除 Redis 会话失败: {e}")

    # ========================================================================
    # 长期记忆操作（MySQL）
    # ========================================================================

    def save_conversation_to_mysql(self, session_id: str, user_id: str,
                                    role: str, content: str,
                                    metadata: Optional[Dict] = None):
        """
        保存对话到 MySQL（长期记忆）

        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 角色
            content: 消息内容
            metadata: 元数据
        """

        try:
            with self._mysql_cursor() as (cursor, connection):
                cursor.execute("""
                    INSERT INTO conversations (session_id, user_id, role, content, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (
                    session_id,
                    user_id,
                    role,
                    content,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None
                ))

                # 更新会话元数据
                cursor.execute("""
                    INSERT INTO session_metadata (session_id, user_id, message_count, start_time, is_active)
                    VALUES (%s, %s, 1, NOW(), 1)
                    ON DUPLICATE KEY UPDATE 
                        message_count = message_count + 1,
                        end_time = NOW(),
                        is_active = 1
                """, (session_id, user_id))

                # 更新用户画像统计
                cursor.execute("""
                    INSERT INTO user_profiles (user_id, conversation_count, last_active_at)
                    VALUES (%s, 1, NOW())
                    ON DUPLICATE KEY UPDATE 
                        conversation_count = conversation_count + 1,
                        last_active_at = NOW()
                """, (user_id,))

                connection.commit()
                logger.debug(f"[MemoryManager] 对话已保存到 MySQL: session={session_id}, user={user_id}")
        except Exception as e:
            logger.error(f"[MemoryManager] 保存对话到 MySQL 失败: {e}")

    def get_conversation_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        从 MySQL 获取用户的历史对话
        
        Args:
            user_id: 用户ID
            limit: 返回记录数量限制
            
        Returns:
            List[Dict]: 对话历史列表
        """
        
        try:
            with self._mysql_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT session_id, role, content, metadata, created_at
                    FROM conversations
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                results = cursor.fetchall()
                
                # 转换为列表并反转顺序（因为查询是倒序）
                results_list = list(results)
                results_list.reverse()
                
                logger.debug(f"[MemoryManager] 从 MySQL 获取 {len(results_list)} 条历史记录: user={user_id}")
                return results_list
        except Exception as e:
            logger.error(f"[MemoryManager] 获取历史对话失败: {e}")
            return []

    def update_user_profile(self, user_id: str, preferences: Optional[Dict] = None,
                            interests: Optional[List[str]] = None):
        """
        更新用户画像

        Args:
            user_id: 用户ID
            preferences: 用户偏好
            interests: 兴趣标签列表
        """

        try:
            with self._mysql_cursor() as (cursor, connection):
                updates = []
                params = []

                if preferences is not None:
                    updates.append("preferences = %s")
                    params.append(json.dumps(preferences, ensure_ascii=False))

                if interests is not None:
                    updates.append("interests = %s")
                    params.append(json.dumps(interests, ensure_ascii=False))

                if updates:
                    params.append(user_id)
                    sql = f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = %s"
                    cursor.execute(sql, params)
                    connection.commit()

                    logger.info(f"[MemoryManager] 用户画像已更新: user={user_id}")
        except Exception as e:
            logger.error(f"[MemoryManager] 更新用户画像失败: {e}")

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户画像

        Args:
            user_id: 用户ID

        Returns:
            Dict: 用户画像数据，不存在则返回 None
        """

        try:
            with self._mysql_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT user_id, preferences, interests, conversation_count, 
                           last_active_at, created_at
                    FROM user_profiles
                    WHERE user_id = %s
                """, (user_id,))

                result = cursor.fetchone()

                if result:
                    # 解析 JSON 字段
                    if result.get('preferences'):
                        result['preferences'] = json.loads(result['preferences'])
                    if result.get('interests'):
                        result['interests'] = json.loads(result['interests'])

                return result
        except Exception as e:
            logger.error(f"[MemoryManager] 获取用户画像失败: {e}")
            return None

    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的活跃会话列表

        Args:
            user_id: 用户ID

        Returns:
            List[Dict]: 会话列表
        """

        try:
            with self._mysql_cursor() as (cursor, _):
                cursor.execute("""
                    SELECT session_id, title, message_count, start_time, end_time
                    FROM session_metadata
                    WHERE user_id = %s AND is_active = 1
                    ORDER BY start_time DESC
                    LIMIT 10
                """, (user_id,))

                return cursor.fetchall()
        except Exception as e:
            logger.error(f"[MemoryManager] 获取活跃会话失败: {e}")
            return []

    # ========================================================================
    # 便捷方法：同时保存到 Redis 和 MySQL
    # ========================================================================

    def save_message(self, session_id: str, user_id: str, role: str,
                     content: str, metadata: Optional[Dict] = None,
                     save_to_long_term: bool = True):
        """
        保存消息（同时写入短期和长期记忆）

        Args:
            session_id: 会话ID
            user_id: 用户ID
            role: 角色
            content: 消息内容
            metadata: 元数据
            save_to_long_term: 是否保存到长期记忆
        """

        # 始终保存到 Redis（短期记忆）
        self.save_message_to_redis(session_id, role, content, metadata)

        # 可选保存到 MySQL（长期记忆）
        if save_to_long_term:
            self.save_conversation_to_mysql(session_id, user_id, role, content, metadata)

    def load_context(self, session_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        加载对话上下文（优先从 Redis，降级到 MySQL）

        Args:
            session_id: 会话ID
            user_id: 用户ID（Redis 缺失时从 MySQL 加载）

        Returns:
            List[Dict]: 消息列表
        """

        # 尝试从 Redis 加载
        messages = self.get_messages_from_redis(session_id)

        if messages:
            return messages

        # Redis 中没有，尝试从 MySQL 加载最近的历史
        if user_id:
            logger.info(f"[MemoryManager] Redis 中无会话数据，从 MySQL 加载历史: session={session_id}")
            history = self.get_conversation_history(user_id, limit=50)

            # 将历史记录加载到 Redis
            for record in history:
                self.save_message_to_redis(
                    session_id,
                    record['role'],
                    record['content'],
                    record.get('metadata')
                )

            return history

        return []
