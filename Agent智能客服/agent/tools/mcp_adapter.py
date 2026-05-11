"""
LangChain MCP 适配器模块

架构说明：
将 MCP 工具转换为 LangChain 兼容的工具格式，
使 LangGraph Agent 能够无缝调用 MCP 工具。

核心流程：
1. 从 mcp_registry 导入客户端类
2. 封装为 LangChain @tool 装饰器格式
3. 提供统一的工具获取接口
"""

from typing import List
from langchain_core.tools import BaseTool, tool
from utilss.logger_handler import logger
from agent.tools.mcp_registry import AMapClient, SearchClient


# ============================================================================
# 初始化客户端（独立实例，不依赖 mcp_server）
# ============================================================================
amap_client = AMapClient()
search_client = SearchClient(provider="tavily")


# ============================================================================
# LangChain 工具封装（桥接层）
# ============================================================================

@tool(description="获取指定城市的实时天气信息（高德地图）")
def mcp_amap_weather(city: str) -> str:
    """
    通过高德地图 API 获取城市天气

    Args:
        city: 城市名称

    Returns:
        str: 格式化天气信息
    """
    logger.info(f"[MCP Adapter]调用 mcp_amap_weather, city={city}")

    weather_data = amap_client.get_weather(city, extensions="all")

    if "error" in weather_data:
        return f"天气查询失败: {weather_data['error']}"

    # 处理 forecasts 数据
    forecasts = weather_data.get("forecasts", [])
    if not forecasts:
        return "未获取到天气信息"
    
    # 取第一个城市的预报（通常是匹配度最高的）
    forecast = forecasts[0]
    casts = forecast.get("casts", [])
    
    if not casts:
        return "未获取到天气预报数据"
    
    info = f"【{forecast.get('province', '')}{forecast.get('city', '')}】天气预报\n"
    info += f"发布时间: {forecast.get('reporttime', '')}\n"

    for day in casts[:3]:  # 显示前3天
        info += (
            f"\n{day.get('date', '')} ({day.get('week', '')}):"
            f"\n  白天: {day.get('dayweather', '')} {day.get('daytemp', '')}°C"
            f"\n  夜间: {day.get('nightweather', '')} {day.get('nighttemp', '')}°C"
            f"\n  风向: {day.get('daywind', '')}{day.get('daypower', '')}级"
        )

    return info


@tool(description="搜索地点兴趣点（高德地图 POI 搜索）")
def mcp_amap_poi_search(keywords: str, city: str = "", poi_type: str = "") -> str:
    """
    搜索附近的地点（餐厅、酒店、景点等）

    Args:
        keywords: 搜索关键词
        city: 城市名称（可选）
        poi_type: POI 类型（可选）

    Returns:
        str: POI 搜索结果
    """
    logger.info(f"[MCP Adapter]调用 mcp_amap_poi_search, keywords={keywords}")

    result = amap_client.search_poi(keywords, city, poi_type)

    if "error" in result:
        return f"POI 搜索失败: {result['error']}"

    count = result.get("count", 0)
    if count == 0:
        return "未找到相关地点"

    pois = result.get("pois", [])[:3]
    info = f"找到 {count} 个相关地点（显示前3个）:\n\n"

    for i, poi in enumerate(pois, 1):
        info += (
            f"{i}. {poi.get('name', '未知')}\n"
            f"   地址: {poi.get('address', '未知')}\n"
            f"   电话: {poi.get('tel', '无')}\n\n"
        )

    return info


@tool(description="执行网络信息检索（通用搜索引擎）")
def mcp_web_search(query: str, num_results: int = 5) -> str:
    """
    搜索互联网获取最新信息

    Args:
        query: 搜索查询
        num_results: 结果数量

    Returns:
        str: 搜索结果摘要
    """
    logger.info(f"[MCP Adapter]调用 mcp_web_search, query={query}")

    results = search_client.search(query, num_results)

    if not results:
        return "未搜索到相关信息"

    info = f"搜索结果（共 {len(results)} 条）:\n\n"
    for i, result in enumerate(results, 1):
        info += (
            f"{i}. {result.get('title', '无标题')}\n"
            f"   摘要: {result.get('content', '')[:200]}...\n"
            f"   链接: {result.get('url', '')}\n\n"
        )

        if result.get("answer") and i == 1:
            info = f"💡 答案: {result['answer']}\n\n" + info

    return info


@tool(description="搜索最新新闻资讯")
def mcp_news_search(topic: str, num_results: int = 3) -> str:
    """
    搜索特定主题的最新新闻

    Args:
        topic: 新闻主题
        num_results: 新闻数量

    Returns:
        str: 新闻摘要
    """
    logger.info(f"[MCP Adapter]调用 mcp_news_search, topic={topic}")

    query = f"{topic} 最新新闻"
    results = search_client.search(query, num_results)

    if not results:
        return f"未搜索到关于 '{topic}' 的新闻"

    info = f"📰 关于 '{topic}' 的最新新闻:\n\n"
    for i, result in enumerate(results, 1):
        info += (
            f"{i}. {result.get('title', '无标题')}\n"
            f"   {result.get('content', '')[:150]}...\n\n"
        )

    return info


# ============================================================================
# 工具注册函数
# ============================================================================

def get_mcp_tools() -> List[BaseTool]:
    """
    获取所有 MCP 工具列表

    Returns:
        List[BaseTool]: LangChain 工具列表
    """
    tools = [
        mcp_amap_weather,
        mcp_amap_poi_search,
        mcp_web_search,
        mcp_news_search,
    ]

    logger.info(f"[MCP Adapter]加载 {len(tools)} 个 MCP 工具")
    return tools


if __name__ == "__main__":
    # 测试工具
    tools = get_mcp_tools()
    for t in tools:
        print(f"工具: {t.name}, 描述: {t.description}")
