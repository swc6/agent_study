# 最简 ReAct Agent 实现

from langchain_community.llms.dashscope import DashScope
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 模拟工具
class SearchTool:
    """搜索工具"""
    
    def run(self, query: str) -> str:
        """模拟搜索功能"""
        search_results = {
            "什么是人工智能": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
            "什么是机器学习": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
            "什么是深度学习": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
            "什么是神经网络": "神经网络是一种模仿人脑神经元结构的计算模型，由多层节点组成，用于处理复杂的数据模式。"
        }
        return search_results.get(query, f"未找到关于'{query}'的信息")

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化工具
search_tool = SearchTool()

# ReAct提示词模板
react_prompt = """
你是一个智能助手，需要通过思考-行动-观察循环来解决问题。

可用工具:
- search: 搜索工具，用于获取信息

使用工具的格式:
思考: [你的思考过程]
行动: [工具名称] [工具参数]
观察: [工具执行结果]

任务: {task}

对话历史:
{chat_history}

请按照上述格式输出你的思考、行动和观察。
"""

# 解析LLM响应
def parse_response(response):
    """解析LLM的响应，提取思考、行动和参数"""
    lines = response.split('\n')
    thought = ""
    action = None
    action_input = ""
    
    for line in lines:
        if line.startswith("思考:"):
            thought = line[3:].strip()
        elif line.startswith("行动:"):
            action_part = line[3:].strip()
            if action_part:
                parts = action_part.split(' ', 1)
                action = parts[0]
                action_input = parts[1] if len(parts) > 1 else ""
    
    return thought, action, action_input

# 执行工具
def execute_tool(action, action_input):
    """执行指定的工具"""
    if action == "search":
        return search_tool.run(action_input)
    else:
        return f"未知工具: {action}"

# ReAct Agent主函数
def react_agent(task, max_iterations=5):
    """ReAct Agent主函数"""
    chat_history = []
    
    print(f"任务: {task}")
    print("=" * 50)
    
    for i in range(max_iterations):
        print(f"\n[迭代 {i+1}]")
        
        # 构建提示词
        prompt = react_prompt.format(
            task=task,
            chat_history='\n'.join(chat_history)
        )
        
        # 调用LLM
        response = llm.invoke(prompt)
        print(f"LLM响应:\n{response}")
        
        # 解析响应
        thought, action, action_input = parse_response(response)
        
        # 如果没有行动，直接返回响应
        if not action:
            print("\n[完成]")
            return response
        
        # 执行工具
        tool_result = execute_tool(action, action_input)
        print(f"工具结果: {tool_result}")
        
        # 更新对话历史
        chat_history.append(f"思考: {thought}")
        chat_history.append(f"行动: {action} {action_input}")
        chat_history.append(f"观察: {tool_result}")
    
    print("\n[达到最大迭代次数]")
    return "未能在指定迭代次数内解决问题"

# 测试
def test_react_agent():
    print("=== 测试最简ReAct Agent ===")
    
    # 测试1: 简单问题
    print("\n测试1: 简单问题")
    task1 = "什么是人工智能？"
    result1 = react_agent(task1)
    print(f"\n最终结果: {result1}")
    
    # 测试2: 复杂问题
    print("\n" + "=" * 70)
    print("测试2: 复杂问题")
    task2 = "什么是深度学习？它与机器学习有什么关系？"
    result2 = react_agent(task2)
    print(f"\n最终结果: {result2}")
    
    # 测试3: 多步骤问题
    print("\n" + "=" * 70)
    print("测试3: 多步骤问题")
    task3 = "什么是深度学习？它使用什么技术？"
    result3 = react_agent(task3)
    print(f"\n最终结果: {result3}")

if __name__ == "__main__":
    test_react_agent()