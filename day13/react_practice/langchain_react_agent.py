# LangChain ReAct Agent 实现

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms.dashscope import DashScope
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义工具
class SearchInput(BaseModel):
    """搜索工具的输入参数"""
    query: str = Field(description="搜索查询语句")

class CalculatorInput(BaseModel):
    """计算器工具的输入参数"""
    expression: str = Field(description="要计算的数学表达式")

class WeatherInput(BaseModel):
    """天气查询工具的输入参数"""
    city: str = Field(description="城市名称")

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
            "什么是神经网络": "神经网络是一种模仿人脑神经元结构的计算模型，由多层节点组成，用于处理复杂的数据模式。"
        }
        return search_results.get(query, f"未找到关于'{query}'的信息")
    
    async def _arun(self, query: str) -> str:
        """异步搜索功能"""
        return self._run(query)

class CalculatorTool(BaseTool):
    """计算器工具"""
    name: str = "calculator"
    description: str = "计算器工具，用于计算数学表达式"
    args_schema: type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """计算数学表达式"""
        try:
            # 安全计算，限制表达式类型
            allowed_chars = "0123456789+-*/() "
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """异步计算数学表达式"""
        return self._run(expression)

class WeatherTool(BaseTool):
    """天气查询工具"""
    name: str = "weather"
    description: str = "天气查询工具，用于获取城市天气"
    args_schema: type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        """获取城市天气"""
        weather_data = {
            "北京": "晴，25℃，微风",
            "上海": "多云，22℃，东风3级",
            "广州": "雨，28℃，南风2级",
            "深圳": "晴，26℃，北风1级",
            "杭州": "阴，23℃，东南风2级"
        }
        return weather_data.get(city, f"未找到{city}的天气信息")
    
    async def _arun(self, city: str) -> str:
        """异步获取城市天气"""
        return self._run(city)

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化工具
search_tool = SearchTool()
calculator_tool = CalculatorTool()
weather_tool = WeatherTool()
tools = [search_tool, calculator_tool, weather_tool]

# 初始化ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 测试
def test_langchain_react_agent():
    print("=== 测试LangChain ReAct Agent ===")
    
    # 测试1: 简单问题
    print("\n测试1: 简单问题")
    task1 = "什么是人工智能？"
    result1 = agent.invoke(task1)
    print(f"\n最终结果: {result1}")
    
    # 测试2: 计算问题
    print("\n" + "=" * 70)
    print("测试2: 计算问题")
    task2 = "123 + 456 * 2等于多少？"
    result2 = agent.invoke(task2)
    print(f"\n最终结果: {result2}")
    
    # 测试3: 天气查询
    print("\n" + "=" * 70)
    print("测试3: 天气查询")
    task3 = "北京今天的天气怎么样？"
    result3 = agent.invoke(task3)
    print(f"\n最终结果: {result3}")
    
    # 测试4: 复杂任务
    print("\n" + "=" * 70)
    print("测试4: 复杂任务")
    task4 = "北京今天的天气怎么样？如果温度超过20度，我需要带多少瓶水？"
    result4 = agent.invoke(task4)
    print(f"\n最终结果: {result4}")

if __name__ == "__main__":
    test_langchain_react_agent()