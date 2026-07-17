from openai import OpenAI
import json
import asyncio
from fastmcp import FastMCPClient  # 导入MCP客户端
import subprocess
import sys

BASE_URL = "https://api-inference.modelscope.cn/v1"
API_KEY = "ms-0202de34-de07-41c8-b487-92f5eee795d2"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

target_text = """在国内，智能生产线故障识别技术取得了显著的突破，主要集中在设备状态监测、故障预测 与诊断系统的开发上。随着工业自动化和信息化水平的提高，越来越多的研究开始关注如何通过 先进的算法和技术手段实现生产线的故障早期预警与自动诊断。例如，某些研究者对实验产线设 备的故障机理进行了深入分析，提出了一种基于长短时记忆网络（LSTM）与双重LSTM（Dual LSTM）混合模型的方案。这种方法在传统LSTM网络的基础上，针对“误差累积”问题进行改进， 从而提升了预测精度与鲁棒性。该模型在处理设备状态变化时，不仅能够实现高效的故障预测， 还能为生产线管理者提供实时的故障预警信息，从而显著降低了故障发生的概率，并提高了生产 的稳定性与安全性。特别是在高风险设备或复杂工艺中，该方法表现出了较强的应用潜力。同时，支持向量机（SVM）作为一种经典的机器学习方法，仍然广泛应用于生产线的故障诊 断任务中。为了克服SVM在高维数据和非线性问题中的局限性，国内研究者通过引入粒子群算法 （PSO）对SVM的参数进行优化，进一步提升了诊断系统的准确性与鲁棒性。例如，某些研究基 于Qt平台开发了火控系统故障预测模型，该模型能够实时监控设备状态并在故障发生之前发出预 警，从而提前采取措施避免生产线停机。该系统的实时性和高效性有效保障了生产线的安全与效 率，为国内智能制造领域的应用提供了宝贵的经验。在国外，生产线故障识别领域的研究也取得了显著进展，尤其是在机器学习、深度学习和数据融合技术的应用上。随着工业4.0时代的到来，越来越多的生产线开始利用智能控制系统和大数 据分析进行故障预警与优化管理。国外研究者提出了基于自适应控制图限值的故障检测方法，该 方法结合了S控制图和历史数据的95%与99%分位数作为上限，能有效应对低方差过程的特点。该方法通过自适应的限值调整，使得生产线的实时监控系统能够更加灵活、精准地检测到故障的发生，特别在钢铁制造、化工等低方差生产环境中得到了广泛应用。这种方法的创新之处在于， 能够在不需要大量计算的情况下，实时有效地检测设备的状态变化，并及时发出故障预警。"""



def start_mcp_server():
    return subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


async def main():

    mcp_process = start_mcp_server()
    mcp_client = FastMCPClient(transport='stdio', process=mcp_process)


    prompt = f"请统计这段文本的中文字数：{target_text}"


    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,

        tools=[{
            "type": "function",
            "function": {
                "name": "count_chinese_chars",
                "description": "统计文本中的中文字符数量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "需要统计的文本"}
                    },
                    "required": ["text"]
                }
            }
        }],
        tool_choice="auto"
    )


    message = response.choices[0].message
    if hasattr(message, 'tool_calls') and message.tool_calls:

        tool_call = message.tool_calls[0].function
        tool_call_params = json.loads(tool_call.arguments)
        print(f"工具名：{tool_call.name}，参数：{tool_call_params}")

        print("\n=== 调用MCP服务执行工具 ===")

        result = await mcp_client.call(
            tool_name=tool_call.name,
            **tool_call_params
        )
        print("MCP服务返回：", result)


        print("\n=== 模型基于工具结果生成最终回答 ===")
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "", "tool_calls": message.tool_calls},
                {"role": "tool", "content": result, "tool_call_id": message.tool_calls[0].id}
            ]
        )
        print("模型最终回答：", final_response.choices[0].message.content.strip())
    else:

        print("模型未调用工具，直接回答：", message.content)


    mcp_process.terminate()


if __name__ == "__main__":
    asyncio.run(main())