# 查询改写与RAG集成

from langchain_community.llms.dashscope import DashScope
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 创建示例向量存储
def create_sample_vectorstore():
    """创建示例向量存储"""
    documents = [
        "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。",
        "强化学习是机器学习的一种方法，通过与环境交互并接收奖励信号来学习最优策略。",
        "监督学习是机器学习的一种方法，使用标记数据来训练模型。",
        "无监督学习是机器学习的一种方法，从无标记数据中学习模式。"
    ]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    return vectorstore

# 查询改写函数
def query_rewrite(query):
    """查询改写"""
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

# 带查询改写的RAG系统
def rag_with_rewrite(query):
    """带查询改写的RAG系统"""
    # 改写查询
    rewritten_query = query_rewrite(query)
    print(f"原始查询: {query}")
    print(f"改写查询: {rewritten_query}")
    
    # 创建向量存储
    vectorstore = create_sample_vectorstore()
    
    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    # 执行检索
    result = qa_chain.invoke(rewritten_query)
    
    return result

# 不带查询改写的RAG系统（用于对比）
def rag_without_rewrite(query):
    """不带查询改写的RAG系统"""
    # 创建向量存储
    vectorstore = create_sample_vectorstore()
    
    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    # 执行检索
    result = qa_chain.invoke(query)
    
    return result

# 比较查询改写的效果
def compare_rewrite_effect():
    """比较查询改写的效果"""
    queries = [
        "什么是AI？",
        "机器学习和深度学习有什么不一样？",
        "我想了解自然语言处理",
        "怎么学习人工智能？"
    ]
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"查询: {query}")
        print("=" * 80)
        
        # 带查询改写的结果
        print("\n1. 带查询改写的结果:")
        result_with_rewrite = rag_with_rewrite(query)
        print(f"答案: {result_with_rewrite['result']}")
        print("来源:")
        for doc in result_with_rewrite['source_documents']:
            print(f"- {doc.page_content[:100]}...")
        
        # 不带查询改写的结果
        print("\n2. 不带查询改写的结果:")
        result_without_rewrite = rag_without_rewrite(query)
        print(f"答案: {result_without_rewrite['result']}")
        print("来源:")
        for doc in result_without_rewrite['source_documents']:
            print(f"- {doc.page_content[:100]}...")

# 测试
def test_rag_integration():
    print("=== 测试查询改写与RAG集成 ===")
    compare_rewrite_effect()

if __name__ == "__main__":
    test_rag_integration()