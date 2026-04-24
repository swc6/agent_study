# 基础工具封装示例

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class WeatherInput(BaseModel):
    """天气查询工具的输入参数"""
    city: str = Field(description="城市名称")

class CalculatorInput(BaseModel):
    """计算器工具的输入参数"""
    expression: str = Field(description="要计算的数学表达式")

class SearchInput(BaseModel):
    """搜索工具的输入参数"""
    query: str = Field(description="搜索查询语句")

class WeatherTool(BaseTool):
    """天气查询工具"""
    name: str = "weather"
    description: str = "获取指定城市的天气信息"
    args_schema: type[BaseModel] = WeatherInput
    
    def _run(self, city: str) -> str:
        """获取城市天气"""
        try:
            # 模拟天气数据
            weather_data = {
                "北京": "晴，25℃，微风",
                "上海": "多云，22℃，东风3级",
                "广州": "雨，28℃，南风2级",
                "深圳": "晴，26℃，北风1级",
                "杭州": "阴，23℃，东南风2级"
            }
            
            if city in weather_data:
                return f"{city}的天气：{weather_data[city]}"
            else:
                return f"未找到{city}的天气信息"
        except Exception as e:
            return f"获取天气失败: {str(e)}"
    
    async def _arun(self, city: str) -> str:
        """异步获取城市天气"""
        return self._run(city)

class CalculatorTool(BaseTool):
    """计算器工具"""
    name: str = "calculator"
    description: str = "用于计算数学表达式的工具"
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

class SearchTool(BaseTool):
    """搜索工具"""
    name: str = "search"
    description: str = "搜索互联网信息的工具"
    args_schema: type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """搜索信息"""
        try:
            # 模拟搜索结果
            search_results = {
                "什么是人工智能?": "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
                "LangChain是什么?": "LangChain是一个用于构建LLM应用的框架，提供了丰富的工具和组件，使开发者能够更轻松地创建复杂的AI应用。",
                "ReAct模式": "ReAct是一种结合推理和行动的Agent架构，核心是思考-行动-观察循环。通过这种模式，Agent能够先思考、再行动、然后观察结果，不断调整策略。",
                "Qwen 3.5": "Qwen 3.5是阿里云开发的大语言模型，具有强大的理解和生成能力，支持多轮对话和复杂任务处理。"
            }
            
            if query in search_results:
                return search_results[query]
            else:
                return f"未找到关于'{query}'的信息"
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """异步搜索信息"""
        return self._run(query)

# 示例用法
def test_basic_tools():
    print("=== 测试基础工具 ===")
    
    # 测试天气工具
    weather_tool = WeatherTool()
    print("\n测试天气工具:")
    result = weather_tool._run("北京")
    print(f"北京天气: {result}")
    
    # 测试计算器工具
    calculator_tool = CalculatorTool()
    print("\n测试计算器工具:")
    result = calculator_tool._run("2 + 3 * 4")
    print(f"计算结果: {result}")
    
    # 测试搜索工具
    search_tool = SearchTool()
    print("\n测试搜索工具:")
    result = search_tool._run("什么是人工智能?")
    print(f"搜索结果: {result}")

if __name__ == "__main__":
    test_basic_tools()