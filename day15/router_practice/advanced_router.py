# 高级路由实现

from langchain_community.llms.dashscope import DashScope
from langchain.prompts import PromptTemplate
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
def advanced_router(query):
    """高级路由实现"""
    prompt = intent_prompt.format(query=query)
    response = llm.invoke(prompt)
    return response.strip()

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

def handle_rag(query):
    """处理需要检索知识库的问题"""
    # 这里可以集成RAG系统
    # 为了演示，我们使用一个简单的实现
    rag_responses = {
        "什么是人工智能": "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "什么是机器学习": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "什么是深度学习": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。"
    }
    
    for key in rag_responses:
        if key in query:
            return rag_responses[key]
    
    # 如果没有找到相关知识，使用LLM生成回答
    prompt = f"""
    请回答以下问题：
    
    {query}
    
    要求：
    1. 回答要准确、全面
    2. 使用简洁明了的语言
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
    intent = advanced_router(query)
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
    elif intent == "rag":
        return handle_rag(query)
    else:
        return handle_default(query)

# 测试
def test_advanced_router():
    print("=== 测试高级路由 ===")
    
    queries = [
        "你好！",
        "123 + 456等于多少？",
        "北京今天的天气怎么样？",
        "什么是人工智能？",
        "如何学习编程？",
        "今天星期几？",
        "机器学习和深度学习有什么区别？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        result = process_query(query)
        print(f"结果: {result}")
        print("-" * 50)

if __name__ == "__main__":
    test_advanced_router()