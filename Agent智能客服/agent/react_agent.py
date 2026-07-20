"""
LangGraph Agent 核心模块

架构说明：
LangGraph 基于状态图（State Graph）的概念，将 Agent 的执行流程建模为一个有向图。
图中的每个节点（Node）执行特定任务，边（Edge）控制流程走向。

核心组件：
1. State（状态）：定义整个 Agent 运行时需要维护的数据结构
2. Node（节点）：执行具体任务的函数，接收状态并返回状态更新
3. Edge（边）：控制节点之间的流转逻辑
4. Graph（图）：由节点和边组成的完整工作流

MCP 集成说明：
- 通过 mcp_adapter 加载高德地图和搜索引擎工具
- MCP 工具与传统工具统一管理，LLM 自主选择调用

记忆系统集成：
- 使用 MemoryManager 管理短期（Redis）和长期（MySQL）记忆
- 支持跨窗口上下文记忆
- 自动保存对话历史到双层存储
"""

from typing import Annotated, TypedDict, Optional
import uuid
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from model.factory import chat_model
from utilss.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.mcp_adapter import get_mcp_tools
from utilss.logger_handler import logger
from utilss.config_handler import agent_conf
from memory.memory_manager import MemoryManager


# ============================================================================
# 第一步：定义状态结构（State Schema）
# ============================================================================
class AgentState(TypedDict):
    """
    Agent 状态定义
    
    messages: 
        - 存储所有对话历史消息（包括用户输入、AI回复、工具调用等）
        - Annotated 指定了消息的合并策略：lambda x, y: x + y 表示新消息追加到列表末尾
    
    report:
        - 布尔标志，标识当前是否处于"报告生成"模式
        - 用于动态切换系统提示词（普通对话 vs 报告生成）
    
    session_id:
        - 会话唯一标识符
        - 用于记忆系统追踪对话上下文
    
    user_id:
        - 用户唯一标识符
        - 用于关联用户画像和历史记录
    """
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    report: bool
    session_id: str
    user_id: str


# ============================================================================
# 第二步：构建 Agent 类
# ============================================================================
class ReactAgent:
    """
    ReAct Agent 实现
    
    ReAct (Reasoning + Acting) 是一种经典的 Agent 架构：
    1. 思考（Reasoning）：LLM 分析用户问题，决定是否需要调用工具
    2. 行动（Acting）：如果需要，调用相应工具获取信息
    3. 观察（Observation）：获取工具返回结果
    4. 重复上述过程，直到能够回答用户问题
    
    LangGraph 通过状态图实现了这个循环流程
    
    MCP 增强：
    - 集成高德地图 MCP 工具（天气、POI 搜索）
    - 集成搜索引擎 MCP 工具（网络检索、新闻搜索）
    
    记忆增强：
    - 集成双层记忆系统（Redis + MySQL）
    - 支持跨窗口上下文记忆
    - 自动保存和加载对话历史
    """
    
    def __init__(self):
        """
        初始化 Agent，构建状态图
        
        构建流程：
        1. 创建工具列表（传统工具 + MCP 工具）
        2. 创建工具节点（ToolNode）：统一管理所有工具的调用
        3. 创建状态图（StateGraph）：定义状态类型
        4. 添加节点（Nodes）：定义图中的处理单元
        5. 设置入口点（Entry Point）：指定图的起始节点
        6. 添加条件边（Conditional Edges）：定义节点间的流转逻辑
        7. 编译图（Compile）：生成可执行的图对象
        8. 初始化记忆管理器
        """
        
        # --------------------------------------------------------------------
        # 2.1 准备工具列表（整合传统工具和 MCP 工具）
        # --------------------------------------------------------------------
        self.traditional_tools = [
            rag_summarize, 
            # get_weather,  # 已禁用：使用 MCP 工具 mcp_amap_weather 替代
            get_user_location, 
            get_user_id,
            get_current_month, 
            fetch_external_data, 
            fill_context_for_report
        ]
        
        # 加载 MCP 工具（如果启用）
        self.mcp_tools = []
        if agent_conf.get("features", {}).get("enable_mcp_tools", True):
            try:
                self.mcp_tools = get_mcp_tools()
                logger.info(f"[ReactAgent]成功加载 {len(self.mcp_tools)} 个 MCP 工具")
            except Exception as e:
                logger.error(f"[ReactAgent]加载 MCP 工具失败: {e}，将仅使用传统工具")
        
        # 合并所有工具
        self.tools = self.traditional_tools + self.mcp_tools
        
        logger.info(f"[ReactAgent]总计加载 {len(self.tools)} 个工具（传统: {len(self.traditional_tools)}, MCP: {len(self.mcp_tools)}）")
        
        # --------------------------------------------------------------------
        # 2.2 创建工具节点（ToolNode）
        # --------------------------------------------------------------------
        self.tool_node = ToolNode(self.tools)
        
        # --------------------------------------------------------------------
        # 2.3 创建状态图构建器
        # --------------------------------------------------------------------
        self.graph_builder = StateGraph(AgentState)
        
        # --------------------------------------------------------------------
        # 2.4 添加节点（Nodes）
        # --------------------------------------------------------------------
        self.graph_builder.add_node("agent", self._call_model)
        self.graph_builder.add_node("tools", self.tool_node)
        
        # --------------------------------------------------------------------
        # 2.5 设置入口点（Entry Point）
        # --------------------------------------------------------------------
        self.graph_builder.set_entry_point("agent")
        
        # --------------------------------------------------------------------
        # 2.6 添加条件边（Conditional Edges）
        # --------------------------------------------------------------------
        self.graph_builder.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # --------------------------------------------------------------------
        # 2.7 添加工具到 Agent 的回边
        # --------------------------------------------------------------------
        self.graph_builder.add_edge("tools", "agent")
        
        # --------------------------------------------------------------------
        # 2.8 编译图（Compile）
        # --------------------------------------------------------------------
        self.graph = self.graph_builder.compile()
        
        # --------------------------------------------------------------------
        # 2.9 初始化记忆管理器
        # --------------------------------------------------------------------
        try:
            self.memory_manager = MemoryManager()
            logger.info("[ReactAgent] 记忆管理器初始化成功")
        except Exception as e:
            logger.error(f"[ReactAgent] 记忆管理器初始化失败: {e}，将不使用记忆功能")
            self.memory_manager = None

    def _call_model(self, state: AgentState):
        """
        Agent 节点：调用 LLM 进行推理
        
        Args:
            state: 当前 Agent 状态，包含消息历史和报告标志
            
        Returns:
            dict: 状态更新，只返回需要更新的字段
        """
        
        messages = state["messages"]
        is_report = state.get("report", False)
        
        # --------------------------------------------------------------------
        # 动态提示词切换
        # --------------------------------------------------------------------
        if is_report:
            system_prompt = load_report_prompts()
        else:
            system_prompt = load_system_prompts()
        
        # --------------------------------------------------------------------
        # 日志记录
        # --------------------------------------------------------------------
        logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息。")
        if messages:
            logger.debug(f"[log_before_model]{type(messages[-1]).__name__} | {messages[-1].content.strip()}")
        
        # --------------------------------------------------------------------
        # 【关键修复】将工具绑定到模型
        # --------------------------------------------------------------------
        model_with_tools = chat_model.bind_tools(self.tools)
        
        # --------------------------------------------------------------------
        # 调用 LLM
        # --------------------------------------------------------------------
        response = model_with_tools.invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        
        # --------------------------------------------------------------------
        # 返回状态更新
        # --------------------------------------------------------------------
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """
        条件判断函数：决定是否继续调用工具
        
        Args:
            state: 当前 Agent 状态
            
        Returns:
            str: "continue" 表示继续调用工具，"end" 表示结束对话
        """
        
        messages = state["messages"]
        last_message = messages[-1]
        
        # --------------------------------------------------------------------
        # 检查是否有工具调用请求
        # --------------------------------------------------------------------
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"[_should_continue]检测到工具调用: {[tc['name'] for tc in last_message.tool_calls]}")
            return "continue"
        
        logger.info(f"[_should_continue]无工具调用，结束对话")
        return "end"

    def execute_stream(self, query: str, session_id: Optional[str] = None, 
                       user_id: Optional[str] = None):
        """
        执行 Agent 并流式返回结果
        
        Args:
            query: 用户的查询字符串
            session_id: 会话ID（可选，不提供则自动生成）
            user_id: 用户ID（可选，用于记忆系统）
            
        Yields:
            str: LLM 响应的文本片段（流式输出）
        """
        
        # --------------------------------------------------------------------
        # 生成或使用提供的会话ID和用户ID
        # --------------------------------------------------------------------
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if not user_id:
            user_id = "anonymous"
        
        logger.info(f"[execute_stream]开始执行查询: {query}, session={session_id}, user={user_id}")
        
        # --------------------------------------------------------------------
        # 从记忆系统加载历史上下文
        # --------------------------------------------------------------------
        historical_messages = []
        if self.memory_manager:
            try:
                context = self.memory_manager.load_context(session_id, user_id)
                
                # 将历史消息转换为 LangChain 消息格式
                for msg in context:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    
                    if role == "user":
                        historical_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        historical_messages.append(AIMessage(content=content))
                    elif role == "tool":
                        metadata = msg.get("metadata", {})
                        tool_name = metadata.get("tool_name", "unknown")
                        historical_messages.append(ToolMessage(content=content, name=tool_name))
                
                logger.info(f"[execute_stream]加载了 {len(historical_messages)} 条历史消息")
            except Exception as e:
                logger.error(f"[execute_stream]加载历史消息失败: {e}")
        
        # --------------------------------------------------------------------
        # 构造初始状态
        # --------------------------------------------------------------------
        input_state = {
            "messages": historical_messages + [HumanMessage(content=query)],
            "report": False,
            "session_id": session_id,
            "user_id": user_id
        }

        # --------------------------------------------------------------------
        # 流式执行图工作流
        # --------------------------------------------------------------------
        step_counter = 0
        full_response = ""

        def _iter_text_chunks(text: str):
            stripped = text.strip()
            if not stripped:
                return

            for raw_line in stripped.splitlines():
                line = raw_line.strip()
                if not line:
                    yield "\n"
                    continue

                start = 0
                while start < len(line):
                    end = min(len(line), start + 40)
                    window = line[start:end]
                    break_at = -1
                    for mark in "。！？!?，,；;：:":
                        pos = window.rfind(mark)
                        if pos > break_at:
                            break_at = pos

                    if break_at >= 15:
                        end = start + break_at + 1

                    yield line[start:end]
                    start = end

                yield "\n"
        
        for event in self.graph.stream(input_state, stream_mode="values"):
            step_counter += 1
            logger.info(f"[execute_stream]步骤 {step_counter}: 收到事件")
            
            messages = event.get("messages", [])
            logger.info(f"[execute_stream]当前消息数量: {len(messages)}")
            
            if messages:
                latest_message = messages[-1]
                logger.info(f"[execute_stream]最新消息类型: {type(latest_message).__name__}")
                
                # ----------------------------------------------------------------
                # 关键修复：只输出 AI 的最终回答，跳过工具调用和工具结果
                # ----------------------------------------------------------------
                
                if isinstance(latest_message, AIMessage):
                    if hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        logger.info(f"[execute_stream]检测到工具调用: {[tc['name'] for tc in latest_message.tool_calls]}")
                        
                        # 保存工具调用到记忆系统
                        if self.memory_manager:
                            try:
                                self.memory_manager.save_message(
                                    session_id=session_id,
                                    user_id=user_id,
                                    role="assistant",
                                    content="[工具调用中...]",
                                    metadata={
                                        "tool_calls": latest_message.tool_calls,
                                        "type": "tool_call"
                                    },
                                    save_to_long_term=agent_conf.get("memory", {}).get("strategy", {}).get("save_to_long_term", True)
                                )
                            except Exception as e:
                                logger.error(f"[execute_stream]保存工具调用失败: {e}")
                        
                        continue
                    
                    if latest_message.content:
                        logger.info(f"[execute_stream]输出 AI 回答")
                        content = latest_message.content
                        
                        if isinstance(content, str):
                            for chunk in _iter_text_chunks(content):
                                full_response += chunk
                                yield chunk
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    for chunk in _iter_text_chunks(item['text']):
                                        full_response += chunk
                                        yield chunk
                
                elif isinstance(latest_message, ToolMessage):
                    logger.info(f"[execute_stream]跳过工具结果: {latest_message.name}")
                    
                    # 保存工具结果到记忆系统
                    if self.memory_manager:
                        try:
                            self.memory_manager.save_message(
                                session_id=session_id,
                                user_id=user_id,
                                role="tool",
                                content=latest_message.content,
                                metadata={
                                    "tool_name": latest_message.name,
                                    "type": "tool_result"
                                },
                                save_to_long_term=agent_conf.get("memory", {}).get("strategy", {}).get("save_to_long_term", True)
                            )
                        except Exception as e:
                            logger.error(f"[execute_stream]保存工具结果失败: {e}")
                    
                    continue
        
        # --------------------------------------------------------------------
        # 保存最终的用户问题和 AI 回答到记忆系统
        # --------------------------------------------------------------------
        if self.memory_manager and full_response:
            try:
                # 保存用户问题
                self.memory_manager.save_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=query,
                    metadata={"type": "question"},
                    save_to_long_term=agent_conf.get("memory", {}).get("strategy", {}).get("save_to_long_term", True)
                )
                
                # 保存 AI 回答
                self.memory_manager.save_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=full_response.strip(),
                    metadata={"type": "answer"},
                    save_to_long_term=agent_conf.get("memory", {}).get("strategy", {}).get("save_to_long_term", True)
                )
                
                logger.info(f"[execute_stream]对话已保存到记忆系统: session={session_id}")
            except Exception as e:
                logger.error(f"[execute_stream]保存对话到记忆系统失败: {e}")


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    agent = ReactAgent()

    # 测试查询：使用 MCP 工具获取真实数据
    for chunk in agent.execute_stream("上海现在天气怎么样？帮我找一家上海世客科技附件的餐厅，最新的扫地机器人技术趋势是什么？"):
        print(chunk, end="", flush=True)
