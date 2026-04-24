# 与Agent集成的路由实现

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms.dashscope import DashScope
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 意图识别提示词
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    请识别以下用户查询的意图类型，返回结果只能是以下之一：
    - chitchat: 闲聊、问候、礼貌用语等
    - professional: 专业知识查询、技术问题等
    - calculation: 数学计算、数值问题等
    - weather: 天气查询
    - rag: 需要检索知识库的问题
    - other: 其他类型
    
    用户查询：{query}
    
    请只返回意图类型，不要添加任何其他内容。
    """
)

# 路由函数
def router(query):
    """路由函数"""
    prompt = intent_prompt.format(query=query)
    response = llm.invoke(prompt)
    return response.strip()

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
            "什么是深度学习": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。"
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

# 初始化工具
search_tool = SearchTool()
calculator_tool = CalculatorTool()
weather_tool = WeatherTool()

# 基于路由的Agent
def routed_agent(query):
    """基于路由的Agent"""
    # 路由
    intent = router(query)
    print(f"识别的意图: {intent}")
    
    # 根据意图选择工具和处理方式
    if intent == "chitchat":
        # 闲聊不需要工具
        responses = [
            "你好！有什么我可以帮助你的吗？",
            "嗨！今天过得怎么样？",
            "你好啊！很高兴为你服务。",
            "嗨，有什么我能为你做的吗？"
        ]
        import random
        return random.choice(responses)
    
    elif intent == "calculation":
        # 计算需要计算器工具
        agent = initialize_agent(
            tools=[calculator_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent.invoke(query)
    
    elif intent == "weather":
        # 天气查询需要天气工具
        agent = initialize_agent(
            tools=[weather_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent.invoke(query)
    
    elif intent == "professional" or intent == "rag":
        # 专业问题和知识库查询需要搜索工具
        agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent.invoke(query)
    
    else:
        # 其他情况，使用所有工具
        agent = initialize_agent(
            tools=[search_tool, calculator_tool, weather_tool],
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent.invoke(query)

# 测试
def test_routed_agent():
    print("=== 测试基于路由的Agent ===")
    
    queries = [
        "你好！",
        "123 + 456等于多少？",
        "北京今天的天气怎么样？",
        "什么是人工智能？",
        "如何学习编程？"
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        print(f"查询: {query}")
        print("=" * 70)
        result = routed_agent(query)
        print(f"\n最终结果: {result}")

if __name__ == "__main__":
    test_routed_agent()