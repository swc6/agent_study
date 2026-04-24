# 基础查询改写实现

from langchain_community.llms.dashscope import DashScope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

def basic_query_rewrite(query):
    """基础查询改写"""
    prompt = f"""
    请将以下用户查询转换为更适合检索系统的专业检索语句：
    
    用户查询：{query}
    
    要求：
    1. 保持查询的核心意图
    2. 使用更专业、更精确的语言
    3. 去除冗余信息
    4. 输出改写后的查询，不要添加任何其他内容
    """
    
    response = llm.invoke(prompt)
    return response.strip()

# 批量查询改写
def batch_rewrite(queries):
    """批量查询改写"""
    results = []
    for query in queries:
        rewritten = basic_query_rewrite(query)
        results.append({
            "original": query,
            "rewritten": rewritten
        })
    return results

# 测试
def test_basic_rewrite():
    print("=== 测试基础查询改写 ===")
    
    queries = [
        "什么是人工智能啊？",
        "机器学习和深度学习有什么不一样？",
        "我想知道怎么学习编程",
        "为什么天空是蓝色的？",
        "Python和Java哪个更好？",
        "如何提高英语口语？",
        "感冒了应该吃什么药？",
        "北京有哪些好玩的地方？"
    ]
    
    results = batch_rewrite(queries)
    
    for result in results:
        print(f"原始查询: {result['original']}")
        print(f"改写查询: {result['rewritten']}")
        print("-" * 50)

if __name__ == "__main__":
    test_basic_rewrite()