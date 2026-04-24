# 基础路由实现

from langchain_community.llms.dashscope import DashScope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

def basic_router(query):
    """基础路由实现"""
    # 闲聊关键词
    chitchat_keywords = ["你好", "嗨", "早上好", "下午好", "晚上好", "再见", "谢谢", "不客气", "你是谁", "你叫什么"]
    
    # 专业问题关键词
    professional_keywords = ["什么是", "如何", "为什么", "原理", "方法", "技术", "算法", "定义", "区别", "应用"]
    
    # 计算相关关键词
    calculation_keywords = ["计算", "等于", "加", "减", "乘", "除", "平方", "平方根", "总和", "平均值"]
    
    # 天气查询关键词
    weather_keywords = ["天气", "温度", "下雨", "下雪", "晴天", "多云"]
    
    # 检查是否包含闲聊关键词
    for keyword in chitchat_keywords:
        if keyword in query:
            return "chitchat"
    
    # 检查是否包含计算关键词
    for keyword in calculation_keywords:
        if keyword in query:
            return "calculation"
    
    # 检查是否包含天气关键词
    for keyword in weather_keywords:
        if keyword in query:
            return "weather"
    
    # 检查是否包含专业问题关键词
    for keyword in professional_keywords:
        if keyword in query:
            return "professional"
    
    # 默认路由
    return "default"

# 处理函数
def handle_chitchat(query):
    """处理闲聊"""
    responses = [
        "你好！有什么我可以帮助你的吗？",
        "嗨！今天过得怎么样？",
        "你好啊！很高兴为你服务。",
        "嗨，有什么我能为你做的吗？"
    ]
    import random
    return random.choice(responses)

def handle_calculation(query):
    """处理计算"""
    # 简单的计算处理
    try:
        # 提取计算表达式
        import re
        expression = re.findall(r'[0-9+\-*/() ]+', query)
        if expression:
            result = eval(expression[0])
            return f"计算结果: {result}"
        else:
            return "抱歉，我无法理解这个计算请求。"
    except Exception as e:
        return f"计算错误: {str(e)}"

def handle_weather(query):
    """处理天气查询"""
    # 提取城市名称
    import re
    cities = re.findall(r'([北京上海广州深圳杭州成都武汉西安南京重庆天津]*)', query)
    city = cities[0] if cities else "北京"
    
    # 模拟天气数据
    weather_data = {
        "北京": "晴，25℃，微风",
        "上海": "多云，22℃，东风3级",
        "广州": "雨，28℃，南风2级",
        "深圳": "晴，26℃，北风1级",
        "杭州": "阴，23℃，东南风2级"
    }
    
    weather = weather_data.get(city, "晴，20℃，微风")
    return f"{city}今天的天气: {weather}"

def handle_professional(query):
    """处理专业问题"""
    prompt = f"""
    请回答以下专业问题：
    
    {query}
    
    要求：
    1. 回答要专业、准确
    2. 内容要详细、全面
    3. 使用简洁明了的语言
    """
    
    response = llm.invoke(prompt)
    return response

def handle_default(query):
    """处理默认情况"""
    prompt = f"""
    请回答以下问题：
    
    {query}
    
    要求：
    1. 回答要准确、全面
    2. 使用简洁明了的语言
    3. 保持友好的语气
    """
    
    response = llm.invoke(prompt)
    return response

# 主处理函数
def process_query(query):
    """处理用户查询"""
    # 路由
    intent = basic_router(query)
    print(f"识别的意图: {intent}")
    
    # 根据意图处理
    if intent == "chitchat":
        return handle_chitchat(query)
    elif intent == "calculation":
        return handle_calculation(query)
    elif intent == "weather":
        return handle_weather(query)
    elif intent == "professional":
        return handle_professional(query)
    else:
        return handle_default(query)

# 测试
def test_basic_router():
    print("=== 测试基础路由 ===")
    
    queries = [
        "你好！",
        "123 + 456等于多少？",
        "北京今天的天气怎么样？",
        "什么是人工智能？",
        "如何学习编程？",
        "今天星期几？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        result = process_query(query)
        print(f"结果: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_basic_router()