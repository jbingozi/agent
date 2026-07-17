
from fastmcp import FastMCP

mcp = FastMCP("计算文本中文数")

@mcp.tool()
def count_chinese_chars(text: str) -> str:

    count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    return f"中文字数统计结果：{count}"

# 3. 运行服务
if __name__ == "__main__":

    mcp.run(transport='stdio')