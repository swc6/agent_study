# 高级查询改写实现

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

# 创建查询改写模板
rewrite_prompt = PromptTemplate(
    input_variables=["query", "domain"],
    template="""
    请将以下{domain}领域的用户查询转换为更适合检索系统的专业检索语句：
    
    用户查询：{query}
    
    要求：
    1. 保持查询的核心意图
    2. 使用{domain}领域的专业术语
    3. 去除冗余信息和口语化表达
    4. 使查询更加精确和具体
    5. 输出改写后的查询，不要添加任何其他内容
    """
)

# 多轮改写提示词
multi_round_prompt = PromptTemplate(
    input_variables=["query", "previous_rewrite", "feedback"],
    template="""
    请根据反馈对之前的查询改写进行优化：
    
    原始查询：{query}
    之前的改写：{previous_rewrite}
    反馈：{feedback}
    
    要求：
    1. 保持查询的核心意图
    2. 结合反馈进行优化
    3. 使查询更加精确和专业
    4. 输出优化后的查询，不要添加任何其他内容
    """
)

def advanced_query_rewrite(query, domain="通用"):
    """高级查询改写"""
    prompt = rewrite_prompt.format(query=query, domain=domain)
    response = llm.invoke(prompt)
    return response.strip()

def multi_round_rewrite(query, feedback):
    """多轮查询改写"""
    # 首先进行基础改写
    first_rewrite = advanced_query_rewrite(query)
    
    # 然后根据反馈进行优化
    prompt = multi_round_prompt.format(
        query=query,
        previous_rewrite=first_rewrite,
        feedback=feedback
    )
    response = llm.invoke(prompt)
    return response.strip()

# 领域特定的查询改写
def domain_specific_rewrite(queries, domains):
    """领域特定的查询改写"""
    results = []
    for query, domain in zip(queries, domains):
        rewritten = advanced_query_rewrite(query, domain)
        results.append({
            "original": query,
            "rewritten": rewritten,
            "domain": domain
        })
    return results

# 测试
def test_advanced_rewrite():
    print("=== 测试高级查询改写 ===")
    
    # 测试领域特定的改写
    print("\n1. 领域特定的查询改写")
    queries = [
        "什么是人工智能？",
        "如何治疗感冒？",
        "怎么投资股票？"
    ]
    domains = ["计算机科学", "医学", "金融"]
    
    results = domain_specific_rewrite(queries, domains)
    for result in results:
        print(f"领域: {result['domain']}")
        print(f"原始查询: {result['original']}")
        print(f"改写查询: {result['rewritten']}")
        print("-" * 50)
    
    # 测试多轮改写
    print("\n2. 多轮查询改写")
    query = "如何学习编程？"
    feedback = "需要更具体，包含初学者的学习路径"
    
    first_rewrite = advanced_query_rewrite(query)
    print(f"原始查询: {query}")
    print(f"第一次改写: {first_rewrite}")
    print(f"反馈: {feedback}")
    
    optimized_rewrite = multi_round_rewrite(query, feedback)
    print(f"优化后的改写: {optimized_rewrite}")
    print("-" * 50)

if __name__ == "__main__":
    test_advanced_rewrite()