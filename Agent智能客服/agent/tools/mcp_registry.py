"""
MCP 工具注册中心模块

架构说明：
MCP (Model Context Protocol) 是 Anthropic 提出的模型上下文协议，
用于标准化 AI 模型与外部数据源和工具的交互方式。

FastMCP 是一个 Python 实现的 MCP 框架，允许我们：
1. 快速创建 MCP 服务器
2. 注册自定义工具和资源
3. 通过标准协议与 LLM 集成

本模块实现：
1. 高德地图 MCP 服务封装
2. 通用搜索引擎 MCP 服务封装
3. 统一工具注册中心
"""

import os
from typing import Optional, Dict, Any, List
from fastmcp import FastMCP
from utilss.config_handler import load_agent_config
from utilss.logger_handler import logger
import requests

# ============================================================================
# 加载 MCP 配置
# ============================================================================
agent_conf = load_agent_config()
mcp_config_path = agent_conf.get("mcp_config_path", "config/mcp.yml")

# 加载 MCP 配置（如果文件存在）
try:
    from utilss.config_handler import yaml
    from utilss.path_tool import get_abs_path

    with open(get_abs_path(mcp_config_path), "r", encoding="utf-8") as f:
        mcp_conf = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    logger.warning(f"[mcp_registry]加载 MCP 配置失败: {e}，使用默认配置")
    mcp_conf = {}


# ============================================================================
# 高德地图 API 客户端封装
# ============================================================================
class AMapClient:
    """
    高德地图 API 客户端

    封装高德地图 Web 服务 API，提供：
    - 天气查询
    - 地理编码（地址转坐标）
    - 逆地理编码（坐标转地址）
    - POI 搜索
    """

    def __init__(self, api_key: str = None):
        """
        初始化高德地图客户端

        Args:
            api_key: 高德地图 API Key，优先从配置读取
        """
        self.api_key = api_key or os.getenv("AMAP_API_KEY") or mcp_conf.get("amap", {}).get("api_key", "")
        self.base_url = mcp_conf.get("amap", {}).get("base_url", "https://restapi.amap.com/v3")

        if not self.api_key:
            logger.warning("[AMapClient]未配置高德地图 API Key，部分功能可能不可用")

    def get_weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        查询城市天气
        
        Args:
            city: 城市名称或城市编码（adcode）
            extensions: all(实况+预报) 或 base(仅实况)
            
        Returns:
            dict: 天气信息
        """
        try:
            endpoint = mcp_conf.get("amap", {}).get("weather", {}).get("endpoint", "/weather/weatherInfo")
            url = f"{self.base_url}{endpoint}"
            
            params = {
                "city": city,
                "key": self.api_key,
                "extensions": extensions
            }
            
            logger.info(f"[AMapClient]请求天气 API: city={city}, extensions={extensions}")
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"[AMapClient]天气 API 响应: status={data.get('status')}, info={data.get('info')}")
            
            if data.get("status") == "1":
                # 返回完整的数据结构，让上层处理
                return data
            else:
                error_msg = data.get("info", "未知错误")
                logger.error(f"[AMapClient]天气查询失败: {error_msg}")
                return {"error": error_msg}

        except Exception as e:
            logger.error(f"[AMapClient]天气查询异常: {str(e)}")
            return {"error": str(e)}

    def geocode(self, address: str, city: str = None) -> Dict[str, Any]:
        """
        地理编码：地址转换为经纬度坐标

        Args:
            address: 详细地址
            city: 城市名称（可选，提高准确性）

        Returns:
            dict: 包含经纬度的地理信息
        """
        try:
            endpoint = mcp_conf.get("amap", {}).get("geocode", {}).get("endpoint", "/geocode/geo")
            url = f"{self.base_url}{endpoint}"

            params = {
                "address": address,
                "key": self.api_key,
                "city": city or ""
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "1" and data.get("geocodes"):
                return data["geocodes"][0]
            else:
                return {"error": data.get("info", "未找到匹配的地理编码")}

        except Exception as e:
            logger.error(f"[AMapClient]地理编码异常: {str(e)}")
            return {"error": str(e)}

    def reverse_geocode(self, longitude: float, latitude: float) -> Dict[str, Any]:
        """
        逆地理编码：经纬度坐标转换为地址

        Args:
            longitude: 经度
            latitude: 纬度

        Returns:
            dict: 包含地址信息的地理信息
        """
        try:
            endpoint = mcp_conf.get("amap", {}).get("regeo", {}).get("endpoint", "/geocode/regeo")
            url = f"{self.base_url}{endpoint}"

            params = {
                "location": f"{longitude},{latitude}",
                "key": self.api_key,
                "pois": 0
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "1":
                return data.get("regeocode", {})
            else:
                return {"error": data.get("info")}

        except Exception as e:
            logger.error(f"[AMapClient]逆地理编码异常: {str(e)}")
            return {"error": str(e)}

    def search_poi(self, keywords: str, city: str = None,
                   types: str = None, page: int = 1) -> Dict[str, Any]:
        """
        POI 搜索：搜索地点兴趣点

        Args:
            keywords: 搜索关键词
            city: 城市限制（可选）
            types: POI 类型（可选，如 "餐饮服务|购物服务"）
            page: 页码

        Returns:
            dict: POI 搜索结果
        """
        try:
            endpoint = mcp_conf.get("amap", {}).get("poi", {}).get("endpoint", "/place/text")
            url = f"{self.base_url}{endpoint}"

            params = {
                "keywords": keywords,
                "key": self.api_key,
                "city": city or "",
                "types": types or mcp_conf.get("amap", {}).get("poi", {}).get("types", ""),
                "page": page,
                "offset": 10
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "1":
                return {
                    "count": int(data.get("count", 0)),
                    "pois": data.get("pois", [])
                }
            else:
                return {"error": data.get("info")}

        except Exception as e:
            logger.error(f"[AMapClient]POI 搜索异常: {str(e)}")
            return {"error": str(e)}


# ============================================================================
# 搜索引擎 API 客户端封装
# ============================================================================
class SearchClient:
    """
    通用搜索引擎客户端

    支持多种搜索引擎后端：
    - Tavily Search（推荐，专为 AI 优化）
    - Serper.dev（Google 搜索代理）
    """

    def __init__(self, provider: str = "tavily", api_key: str = None):
        """
        初始化搜索引擎客户端

        Args:
            provider: 搜索引擎提供商 ("tavily" 或 "serper")
            api_key: API Key
        """
        self.provider = provider
        self.api_key = api_key

        # 从配置加载
        if not self.api_key:
            if provider == "tavily":
                self.api_key = os.getenv("TAVILY_API_KEY") or mcp_conf.get("search", {}).get("tavily", {}).get("api_key", "")
            elif provider == "serper":
                self.api_key = os.getenv("SERPER_API_KEY") or mcp_conf.get("search", {}).get("serper", {}).get("api_key", "")

    def search(self, query: str, num_results: int = 5,
               search_depth: str = "basic") -> List[Dict[str, Any]]:
        """
        执行网络搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数量
            search_depth: 搜索深度 ("basic" 或 "advanced")

        Returns:
            list: 搜索结果列表
        """
        if self.provider == "tavily":
            return self._tavily_search(query, num_results, search_depth)
        elif self.provider == "serper":
            return self._serper_search(query, num_results)
        else:
            logger.error(f"[SearchClient]不支持的搜索引擎: {self.provider}")
            return []

    def _tavily_search(self, query: str, num_results: int = 5,
                       search_depth: str = "basic") -> List[Dict[str, Any]]:
        """Tavily 搜索实现"""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)

            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=num_results,
                include_answer=mcp_conf.get("search", {}).get("tavily", {}).get("include_answer", True),
                include_raw_content=mcp_conf.get("search", {}).get("tavily", {}).get("include_raw_content", False)
            )

            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })

            # 如果有答案，添加到第一个结果
            if response.get("answer"):
                if results:
                    results[0]["answer"] = response["answer"]

            return results

        except ImportError:
            logger.error("[SearchClient]未安装 tavily-python，请运行: pip install tavily-python")
            return []
        except Exception as e:
            logger.error(f"[SearchClient]Tavily 搜索异常: {str(e)}")
            return []

    def _serper_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Serper.dev 搜索实现"""
        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "num": num_results
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for organic in data.get("organic", []):
                results.append({
                    "title": organic.get("title", ""),
                    "url": organic.get("link", ""),
                    "content": organic.get("snippet", ""),
                    "position": organic.get("position", 0)
                })

            return results

        except Exception as e:
            logger.error(f"[SearchClient]Serper 搜索异常: {str(e)}")
            return []


# ============================================================================
# 创建 FastMCP 服务器
# ============================================================================
def create_mcp_server() -> FastMCP:
    """
    创建并配置 FastMCP 服务器

    注册所有 MCP 工具和资源

    Returns:
        FastMCP: 配置好的 MCP 服务器实例
    """

    # 初始化客户端
    amap_client = AMapClient()
    search_client = SearchClient(provider="tavily")

    # 创建 MCP 服务器
    server_config = mcp_conf.get("server", {})
    mcp_server = FastMCP(
        name=server_config.get("name", "agent-mcp-gateway"),
        instructions=server_config.get("description", "统一 MCP 聚合网关")
    )

    # ------------------------------------------------------------------------
    # 注册高德地图工具
    # ------------------------------------------------------------------------

    @mcp_server.tool()
    def amap_get_weather(city: str) -> str:
        """
        获取指定城市的实时天气信息

        Args:
            city: 城市名称（如 "杭州"、"北京"）

        Returns:
            str: 格式化的天气信息
        """
        logger.info(f"[MCP Tool]调用 amap_get_weather, city={city}")

        weather_data = amap_client.get_weather(city, extensions="all")

        if "error" in weather_data:
            return f"天气查询失败: {weather_data['error']}"

        # 格式化天气信息
        if "lives" in weather_data:
            # 实况天气
            live = weather_data["lives"][0] if weather_data["lives"] else {}
            return (
                f"【{live.get('province', '')}{live.get('city', '')}】实时天气\n"
                f"天气: {live.get('weather', '未知')}\n"
                f"温度: {live.get('temperature', '未知')}°C\n"
                f"湿度: {live.get('humidity', '未知')}%\n"
                f"风向: {live.get('winddirection', '未知')}{live.get('windpower', '')}级\n"
                f"发布时间: {live.get('reporttime', '未知')}"
            )
        elif "forecasts" in weather_data:
            # 天气预报
            forecast = weather_data["forecasts"][0] if weather_data["forecasts"] else {}
            casts = forecast.get("casts", [])
            info = f"【{forecast.get('province', '')}{forecast.get('city', '')}】天气预报\n"

            for day in casts[:3]:  # 只显示前3天
                info += (
                    f"\n{day.get('date', '')} ({day.get('week', '')}):\n"
                    f"  白天: {day.get('dayweather', '')} {day.get('daytemp', '')}°C\n"
                    f"  夜间: {day.get('nightweather', '')} {day.get('nighttemp', '')}°C\n"
                    f"  风向: {day.get('daywind', '')}{day.get('daypower', '')}级"
                )

            return info

        return "未获取到天气信息"

    @mcp_server.tool()
    def amap_geocode(address: str, city: str = "") -> str:
        """
        将地址转换为经纬度坐标

        Args:
            address: 详细地址
            city: 城市名称（可选，提高准确性）

        Returns:
            str: 经纬度坐标和相关信息
        """
        logger.info(f"[MCP Tool]调用 amap_geocode, address={address}, city={city}")

        result = amap_client.geocode(address, city)

        if "error" in result:
            return f"地理编码失败: {result['error']}"

        return (
            f"地址: {result.get('formatted_address', '')}\n"
            f"经度: {result.get('location', '').split(',')[0] if result.get('location') else '未知'}\n"
            f"纬度: {result.get('location', '').split(',')[1] if result.get('location') else '未知'}\n"
            f"置信度: {result.get('level', '未知')}"
        )

    @mcp_server.tool()
    def amap_reverse_geocode(longitude: float, latitude: float) -> str:
        """
        将经纬度坐标转换为具体地址

        Args:
            longitude: 经度
            latitude: 纬度

        Returns:
            str: 详细地址信息
        """
        logger.info(f"[MCP Tool]调用 amap_reverse_geocode, lon={longitude}, lat={latitude}")

        result = amap_client.reverse_geocode(longitude, latitude)

        if "error" in result:
            return f"逆地理编码失败: {result['error']}"

        address_component = result.get("addressComponent", {})
        return (
            f"地址: {result.get('formatted_address', '')}\n"
            f"省份: {address_component.get('province', '')}\n"
            f"城市: {address_component.get('city', '')}\n"
            f"区县: {address_component.get('district', '')}\n"
            f"街道: {address_component.get('township', '')}"
        )

    @mcp_server.tool()
    def amap_search_poi(keywords: str, city: str = "", poi_type: str = "") -> str:
        """
        搜索地点兴趣点（POI）

        Args:
            keywords: 搜索关键词（如 "餐厅"、"酒店"）
            city: 城市限制（可选）
            poi_type: POI 类型（可选，如 "餐饮服务"、"住宿服务"）

        Returns:
            str: POI 搜索结果列表
        """
        logger.info(f"[MCP Tool]调用 amap_search_poi, keywords={keywords}, city={city}")

        result = amap_client.search_poi(keywords, city, poi_type)

        if "error" in result:
            return f"POI 搜索失败: {result['error']}"

        count = result.get("count", 0)
        if count == 0:
            return "未找到相关地点"

        pois = result.get("pois", [])[:5]  # 最多返回5个

        info = f"找到 {count} 个相关地点（显示前5个）:\n\n"
        for i, poi in enumerate(pois, 1):
            info += (
                f"{i}. {poi.get('name', '未知')}\n"
                f"   地址: {poi.get('address', '未知')}\n"
                f"   电话: {poi.get('tel', '无')}\n"
                f"   评分: {poi.get('biz_ext', {}).get('rating', '无')}\n"
                f"   距离: {poi.get('distance', '未知')}米\n\n"
            )

        return info

    # ------------------------------------------------------------------------
    # 注册搜索引擎工具
    # ------------------------------------------------------------------------

    @mcp_server.tool()
    def web_search(query: str, num_results: int = 5) -> str:
        """
        执行网络信息检索

        Args:
            query: 搜索查询
            num_results: 返回结果数量（默认5条）

        Returns:
            str: 搜索结果摘要
        """
        logger.info(f"[MCP Tool]调用 web_search, query={query}")

        results = search_client.search(query, num_results)

        if not results:
            return "未搜索到相关信息"

        info = f"搜索结果（共 {len(results)} 条）:\n\n"
        for i, result in enumerate(results, 1):
            info += (
                f"{i}. {result.get('title', '无标题')}\n"
                f"   链接: {result.get('url', '')}\n"
                f"   摘要: {result.get('content', '')[:200]}...\n\n"
            )

            # 如果有直接答案，优先展示
            if result.get("answer") and i == 1:
                info = f"💡 直接答案:\n{result['answer']}\n\n" + info

        return info

    @mcp_server.tool()
    def news_search(topic: str, num_results: int = 3) -> str:
        """
        搜索最新新闻资讯

        Args:
            topic: 新闻主题
            num_results: 返回新闻数量

        Returns:
            str: 新闻摘要
        """
        logger.info(f"[MCP Tool]调用 news_search, topic={topic}")

        # 添加时间限定词进行搜索
        query = f"{topic} 最新新闻"
        results = search_client.search(query, num_results)

        if not results:
            return f"未搜索到关于 '{topic}' 的最新新闻"

        info = f"📰 关于 '{topic}' 的最新新闻:\n\n"
        for i, result in enumerate(results, 1):
            info += (
                f"{i}. {result.get('title', '无标题')}\n"
                f"   来源: {result.get('url', '')}\n"
                f"   摘要: {result.get('content', '')[:150]}...\n\n"
            )

        return info

    # ------------------------------------------------------------------------
    # 注册资源（可选：提供静态信息）
    # ------------------------------------------------------------------------

    @mcp_server.resource("weather://help")
    def weather_help() -> str:
        """天气查询帮助文档"""
        return """
        # 天气查询工具使用说明

        ## 可用工具
        - `amap_get_weather(city)`: 查询城市天气
        - `amap_geocode(address, city)`: 地址转坐标
        - `amap_reverse_geocode(lon, lat)`: 坐标转地址
        - `amap_search_poi(keywords, city, type)`: 搜索地点

        ## 示例
        - 查询杭州天气: amap_get_weather("杭州")
        - 搜索附近餐厅: amap_search_poi("餐厅", "杭州", "餐饮服务")
        """

    @mcp_server.resource("search://help")
    def search_help() -> str:
        """搜索引擎工具使用说明"""
        return """
        # 搜索引擎工具使用说明

        ## 可用工具
        - `web_search(query, num_results)`: 通用网络搜索
        - `news_search(topic, num_results)`: 新闻搜索

        ## 示例
        - 搜索技术文章: web_search("Python FastMCP 教程")
        - 查询最新新闻: news_search("人工智能")
        """

    # 修复：使用正确的 API 获取工具数量
    try:
        # FastMCP 3.x 版本的正确方式
        tool_count = len(mcp_server._tool_manager.list_tools()) if hasattr(mcp_server, '_tool_manager') else 6
        logger.info(f"[MCP Registry]成功注册 {tool_count} 个 MCP 工具")
    except AttributeError:
        logger.info(f"[MCP Registry]成功注册 6 个 MCP 工具（高德4个 + 搜索2个）")

    return mcp_server


# ============================================================================
# 导出全局 MCP 服务器实例（延迟初始化，避免导入时执行）
# ============================================================================
_mcp_server_instance = None

def get_mcp_server() -> FastMCP:
    """获取或创建 MCP 服务器实例（单例模式）"""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        _mcp_server_instance = create_mcp_server()
    return _mcp_server_instance


if __name__ == "__main__":
    # 测试运行 MCP 服务器
    logger.info("[MCP Registry]启动 MCP 服务器...")
    server = get_mcp_server()
    server.run()

