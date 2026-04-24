# 带记忆的 ReAct Agent 实现

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms.dashscope import DashScope
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义工具
class SearchInput(BaseModel):
    """搜索工具的输入参数"""
    query: str = Field(description="搜索查询语句")

class SearchTool(BaseTool):
    """搜索工具"""
    name: str = "search"
    description: str = "搜索工具，用于获取信息"
    args_schema: type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """模拟搜索功能"""
        search_results = {
            "什么是人工智能": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
            "什么是机器学习": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
            "什么是深度学习": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
            "什么是神经网络": "神经网络是一种模仿人脑神经元结构的计算模型，由多层节点组成，用于处理复杂的数据模式。",
            "人工智能的应用": "人工智能的应用领域包括医疗、金融、教育、交通、娱乐等多个方面。",
            "机器学习的算法": "机器学习的算法包括监督学习、无监督学习、强化学习等。"
        }
        return search_results.get(query, f"未找到关于'{query}'的信息")
    
    async def _arun(self, query: str) -> str:
        """异步搜索功能"""
        return self._run(query)

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化记忆
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化工具
search_tool = SearchTool()
tools = [search_tool]

# 初始化ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# 测试多轮对话
def test_memory_react_agent():
    print("=== 测试带记忆的ReAct Agent ===")
    
    # 第一轮
    print("\n测试1: 第一轮对话")
    task1 = "什么是人工智能？"
    result1 = agent.invoke(task1)
    print(f"\n最终结果: {result1}")
    
    # 第二轮
    print("\n" + "=" * 70)
    print("测试2: 第二轮对话")
    task2 = "它有哪些应用领域？"
    result2 = agent.invoke(task2)
    print(f"\n最终结果: {result2}")
    
    # 第三轮
    print("\n" + "=" * 70)
    print("测试3: 第三轮对话")
    task3 = "它与机器学习有什么关系？"
    result3 = agent.invoke(task3)
    print(f"\n最终结果: {result3}")
    
    # 第四轮
    print("\n" + "=" * 70)
    print("测试4: 第四轮对话")
    task4 = "机器学习有哪些算法？"
    result4 = agent.invoke(task4)
    print(f"\n最终结果: {result4}")

if __name__ == "__main__":
    test_memory_react_agent()