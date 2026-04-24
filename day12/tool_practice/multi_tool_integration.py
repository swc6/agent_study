# 多工具集成示例

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms.dashscope import DashScope
from basic_tools import WeatherTool, CalculatorTool, SearchTool
from rag_tool import RAGTool, create_sample_vectorstore
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class MultiToolAgent:
    """多工具智能体"""
    
    def __init__(self):
        """初始化多工具智能体"""
        # 初始化LLM
        self.llm = DashScope(
            model="qwen-plus",
            temperature=0.7
        )
        
        # 初始化工具
        self.weather_tool = WeatherTool()
        self.calculator_tool = CalculatorTool()
        self.search_tool = SearchTool()
        
        # 创建RAG工具
        vectorstore = create_sample_vectorstore()
        self.rag_tool = RAGTool(vectorstore)
        
        # 工具列表
        self.tools = [
            self.weather_tool,
            self.calculator_tool,
            self.search_tool,
            self.rag_tool
        ]
        
        # 初始化智能体
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        print("多工具智能体初始化成功！")
    
    def run(self, query: str) -> str:
        """运行智能体"""
        try:
            result = self.agent.invoke(query)
            return result
        except Exception as e:
            return f"运行失败: {str(e)}"

class ToolSelector:
    """工具选择器"""
    
    def __init__(self):
        """初始化工具选择器"""
        # 初始化工具
        self.weather_tool = WeatherTool()
        self.calculator_tool = CalculatorTool()
        self.search_tool = SearchTool()
        
        # 创建RAG工具
        vectorstore = create_sample_vectorstore()
        self.rag_tool = RAGTool(vectorstore)
        
        # 工具映射
        self.tool_map = {
            "weather": [self.weather_tool],
            "calculation": [self.calculator_tool],
            "search": [self.search_tool],
            "knowledge": [self.rag_tool],
            "all": [self.weather_tool, self.calculator_tool, self.search_tool, self.rag_tool]
        }
    
    def get_agent(self, task_type: str) -> initialize_agent:
        """根据任务类型返回智能体"""
        # 初始化LLM
        llm = DashScope(
            model="qwen-plus",
            temperature=0.7
        )
        
        # 获取工具
        tools = self.tool_map.get(task_type, self.tool_map["all"])
        
        # 初始化智能体
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent

# 示例用法
def test_multi_tool_agent():
    print("=== 测试多工具智能体 ===")
    
    # 初始化多工具智能体
    agent = MultiToolAgent()
    
    # 测试1: 天气查询
    print("\n测试1: 天气查询")
    result = agent.run("北京今天的天气怎么样？")
    print(f"结果: {result}")
    
    # 测试2: 计算
    print("\n测试2: 计算")
    result = agent.run("123 + 456 * 2等于多少？")
    print(f"结果: {result}")
    
    # 测试3: 搜索
    print("\n测试3: 搜索")
    result = agent.run("什么是ReAct模式？")
    print(f"结果: {result}")
    
    # 测试4: 知识库查询
    print("\n测试4: 知识库查询")
    result = agent.run("什么是人工智能？")
    print(f"结果: {result}")
    
    # 测试5: 复杂任务
    print("\n测试5: 复杂任务")
    result = agent.run("北京今天的天气怎么样？如果温度超过20度，我需要带多少瓶水？")
    print(f"结果: {result}")

def test_tool_selector():
    print("\n=== 测试工具选择器 ===")
    
    # 初始化工具选择器
    selector = ToolSelector()
    
    # 测试天气任务
    print("\n测试天气任务")
    weather_agent = selector.get_agent("weather")
    result = weather_agent.invoke("上海今天的天气怎么样？")
    print(f"结果: {result}")
    
    # 测试知识任务
    print("\n测试知识任务")
    knowledge_agent = selector.get_agent("knowledge")
    result = knowledge_agent.invoke("什么是机器学习？")
    print(f"结果: {result}")

if __name__ == "__main__":
    test_multi_tool_agent()
    test_tool_selector()