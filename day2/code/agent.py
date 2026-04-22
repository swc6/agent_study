import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

import dashscope
from dashscope import Generation

# 配置dashscope API密钥和base_url
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_url = os.getenv("DASHSCOPE_BASE_URL")

# 定义工具
def search(query: str) -> str:
    """
    用于搜索信息的工具。
    """
    search_results = {
        "什么是Agent?": "Agent是一种能够自主决策、执行任务并与环境交互的智能系统。",
        "LangChain是什么?": "LangChain是一个用于构建LLM应用的框架。",
        "ReAct模式": "ReAct是一种结合推理和行动的Agent架构，核心是思考-行动-观察循环。",
        "Qwen": "Qwen是阿里云开发的大语言模型系列。"
    }
    return search_results.get(query, f"未找到关于'{query}'的信息")

def calculate(expression: str) -> str:
    """
    用于计算数学表达式的工具。
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具注册表
TOOLS = {
    "search": search,
    "calculate": calculate
}

def qwen_chat(messages, temperature=0.7, top_p=0.9):
    """
    调用Qwen模型的函数
    """
    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        result_format='message'
    )
    if response.status_code == 200:
        return response.output['choices'][0]['message']['content']
    else:
        return f"调用失败: {response.message}"

def parse_action(response):
    """解析响应中的行动"""
    lines = response.split("\n")
    action = None
    for line in lines:
        if line.startswith("行动:"):
            action_str = line[3:].strip()
            parts = action_str.split()
            if parts:
                tool_name = parts[0]
                tool_args = " ".join(parts[1:]) if len(parts) > 1 else ""
                action = (tool_name, tool_args)
            break
    return action

def execute_tool(tool_name, tool_args):
    """执行工具"""
    if tool_name in TOOLS:
        try:
            return TOOLS[tool_name](tool_args)
        except Exception as e:
            return f"工具执行错误: {str(e)}"
    else:
        return f"未知工具: {tool_name}"

def react_agent(query, max_iterations=10):
    """
    ReAct Agent实现
    """
    system_prompt = """你是一个智能助手，需要通过思考-行动-观察循环来解决问题。

可用工具:
- search: 搜索工具，用于搜索信息
- calculate: 计算工具，用于计算数学表达式

使用工具的格式:
思考: [你的思考过程]
行动: [工具名称] [工具参数]
观察: [工具执行结果]

请按照上述格式输出你的思考、行动和观察。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    for i in range(max_iterations):
        # 调用LLM
        response = qwen_chat(messages)

        # 解析行动
        action = parse_action(response)

        if not action:
            # 没有行动，直接返回响应
            return response

        tool_name, tool_args = action

        # 执行工具
        tool_result = execute_tool(tool_name, tool_args)

        # 添加到消息历史
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"观察: {tool_result}"})

        print(f"\n[迭代 {i+1}]")
        print(f"LLM响应: {response}")
        print(f"执行工具: {tool_name}")
        print(f"工具结果: {tool_result}")

    return response

def test_agent():
    print("=== 测试Agent ===")

    # 测试搜索功能
    print("\n测试1: 搜索ReAct模式")
    query1 = "什么是ReAct模式？"
    response1 = react_agent(query1)
    print("\n最终响应:", response1)

    # 测试计算功能
    print("\n\n测试2: 计算数学表达式")
    query2 = "计算 123 + 456 * 2"
    response2 = react_agent(query2)
    print("\n最终响应:", response2)

    # 测试多轮对话
    print("\n\n测试3: 多轮对话")
    query3 = "什么是Qwen？它有什么特点？"
    response3 = react_agent(query3)
    print("\n最终响应:", response3)

if __name__ == "__main__":
    test_agent()