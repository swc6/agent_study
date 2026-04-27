# 答案校验示例

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

# 生成回答
def generate_answer(question, vectorstore, llm):
    """生成回答"""
    # 检索相关文档
    docs = retrieve_documents(question, vectorstore)
    
    # 构建上下文
    context = ""
    for doc in docs:
        context += doc.page_content + "\n"
    
    # 生成回答
    prompt = f"""
    请根据以下上下文回答问题：
    
    上下文：
    {context}
    
    问题：
    {question}
    
    要求：
    1. 基于上下文回答问题
    2. 保持回答简洁、准确
    """
    
    answer = llm.invoke(prompt)
    return answer, context

# 校验回答
def validate_answer(question, answer, context, llm):
    """校验回答"""
    prompt = f"""
    请校验以下回答是否准确回答了问题，并基于提供的上下文：
    
    问题：
    {question}
    
    回答：
    {answer}
    
    上下文：
    {context}
    
    校验标准：
    1. 回答是否直接回答了问题
    2. 回答是否与上下文一致
    3. 回答是否包含错误信息
    4. 回答是否有上下文支持
    
    请给出校验结果（"通过"或"不通过"），并简要说明理由。如果不通过，请提供修正建议。
    """
    
    validation = llm.invoke(prompt)
    return validation

# 测试答案校验
def test_validation():
    """测试答案校验"""
    print("=== 测试答案校验 ===")
    
    # 创建向量存储
    vectorstore = create_sample_vectorstore()
    
    # 测试查询
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么关系？",
        "人工智能有哪些应用领域？",
        "什么是自然语言处理？"
    ]
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"查询: {query}")
        print("=" * 80)
        
        # 生成回答
        answer, context = generate_answer(query, vectorstore, llm)
        
        print("\n上下文:")
        print(context)
        
        print("\n回答:")
        print(answer)
        
        # 校验回答
        validation = validate_answer(query, answer, context, llm)
        
        print("\n校验结果:")
        print(validation)

if __name__ == "__main__":
    test_validation()