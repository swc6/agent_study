# 幻觉抑制示例

from langchain_community.llms.dashscope import DashScope
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
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
        "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。"
    ]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    return vectorstore

# 检索相关文档
def retrieve_documents(query, vectorstore, k=3):
    """检索相关文档"""
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# 创建反幻觉提示词
def create_anti_hallucination_prompt(question, context):
    """创建反幻觉提示词"""
    prompt = f"""
    请根据以下上下文回答问题，确保回答的每个部分都有上下文支持：
    
    上下文：
    {context}
    
    问题：
    {question}
    
    要求：
    1. 只基于上下文回答问题，不要添加任何上下文之外的信息
    2. 如果上下文无法回答问题，请明确表示"根据提供的上下文无法回答此问题"
    3. 对于不确定的信息，明确表示不确定
    4. 保持回答简洁、准确
    """
    return prompt

# 生成带幻觉抑制的回答
def generate_answer_with_anti_hallucination(question, vectorstore, llm):
    """生成带幻觉抑制的回答"""
    # 检索相关文档
    docs = retrieve_documents(question, vectorstore)
    
    # 构建上下文
    context = ""
    for i, doc in enumerate(docs):
        context += f"文档 {i+1}: {doc.page_content}\n"
    
    # 生成回答
    prompt = create_anti_hallucination_prompt(question, context)
    answer = llm.invoke(prompt)
    
    return answer, context

# 测试幻觉抑制
def test_anti_hallucination():
    """测试幻觉抑制"""
    print("=== 测试幻觉抑制 ===")
    
    # 创建向量存储
    vectorstore = create_sample_vectorstore()
    
    # 测试查询
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么关系？",
        "人工智能有哪些应用领域？",
        "什么是自然语言处理？",
        "人工智能的未来发展趋势是什么？"  # 这个问题上下文可能无法完全回答
    ]
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"查询: {query}")
        print("=" * 80)
        
        # 生成回答
        answer, context = generate_answer_with_anti_hallucination(query, vectorstore, llm)
        
        print("\n上下文:")
        print(context)
        
        print("\n回答:")
        print(answer)

if __name__ == "__main__":
    test_anti_hallucination()