"""
Agent 工具函数模块

工具（Tools）是 Agent 与外部世界交互的桥梁。
LLM 本身无法访问实时数据或执行操作，通过工具可以：
- 检索知识库（RAG）
- 查询天气、位置等实时信息
- 访问外部数据库
- 触发特定的业务逻辑
- 调用 MCP 服务（高德地图、搜索引擎）

每个工具都是一个普通的 Python 函数，通过 @tool 装饰器注册，
LLM 可以根据函数的 description 自主决定何时调用。
"""

import os
import functools
from utilss.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utilss.config_handler import agent_conf
from utilss.path_tool import get_abs_path

# ============================================================================
# 初始化 RAG 服务
# ============================================================================
rag = RagSummarizeService()

# ============================================================================
# 模拟数据（实际项目中应从数据库或 API 获取）
# ============================================================================
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


# ============================================================================
# 工具 1：RAG 知识检索
# ============================================================================
@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """
    RAG 检索工具
    
    当用户的问题需要专业知识时，LLM 会调用此工具从向量数据库中检索相关文档。
    
    Args:
        query: 用户的查询字符串
        
    Returns:
        str: 检索到的相关知识总结
    """
    return rag.rag_summarize(query)


# ============================================================================
# 工具 2：天气查询（保留原有模拟实现作为降级方案）
# ============================================================================
@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """
    天气查询工具（模拟实现 - 降级方案）
    
    【注意】优先使用 MCP 工具 mcp_amap_weather 获取真实天气
    此工具作为 MCP 不可用时的降级方案
    
    Args:
        city: 城市名称
        
    Returns:
        str: 天气信息描述
    """
    logger.warning(f"[get_weather]使用模拟天气数据（建议启用 MCP 工具）")
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


# ============================================================================
# 工具 3：获取用户位置
# ============================================================================
@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    用户位置获取工具（模拟实现）
    
    Returns:
        str: 用户所在城市名称
    """
    return random.choice(["深圳", "合肥", "杭州"])


# ============================================================================
# 工具 4：获取用户 ID
# ============================================================================
@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """
    用户 ID 获取工具（模拟实现）
    
    Returns:
        str: 用户 ID
    """
    return random.choice(user_ids)


# ============================================================================
# 工具 5：获取当前月份
# ============================================================================
@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """
    当前月份获取工具（模拟实现）
    
    Returns:
        str: 当前月份（格式：YYYY-MM）
    """
    return random.choice(month_arr)


# ============================================================================
# 辅助函数：加载外部数据
# ============================================================================
def generate_external_data():
    """
    从 CSV 文件加载外部数据到内存缓存
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


# ============================================================================
# 工具 6：获取外部数据
# ============================================================================
@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部数据查询工具
    
    Args:
        user_id: 用户 ID
        month: 月份（格式：YYYY-MM）
        
    Returns:
        str: 用户在该月份的使用记录，未找到则返回空字符串
    """
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


# ============================================================================
# 工具 7：触发报告生成模式
# ============================================================================
@tool(description="无入参，返回确认信息。调用后会将对话切换到报告生成模式，后续使用专门的报告提示词")
def fill_context_for_report() -> str:
    """
    报告生成模式切换工具
    
    Returns:
        str: 确认信息
    """
    logger.info("[fill_context_for_report]报告生成模式已激活")
    return "报告生成模式已激活，系统将使用专门的报告提示词进行回答"
