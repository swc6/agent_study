# 引用溯源示例

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

# 生成带引用的回答
def generate_answer_with_citations(question, vectorstore, llm):
    """生成带引用的回答"""
    # 检索相关文档
    docs = retrieve_documents(question, vectorstore)
    
    # 构建上下文
    context = ""
    sources = []
    for i, doc in enumerate(docs):
        context += f"[{i+1}] {doc.page_content}\n"
        sources.append(f"文档 {i+1}")
    
    # 生成带引用的回答
    prompt = f"""
    请根据以下上下文回答问题，并在回答中使用引用标记（如[1]）指向相关信息的来源：
    
    上下文：
    {context}
    
    问题：
    {question}
    
    要求：
    1. 只基于上下文回答问题，不要添加任何上下文之外的信息
    2. 在回答中使用引用标记
    3. 保持回答简洁、准确
    4. 最后列出所有引用的来源
    """
    
    answer = llm.invoke(prompt)
    
    # 添加来源列表
    sources_list = "\n\n参考来源：\n"
    for i, source in enumerate(sources):
        sources_list += f"[{i+1}] {source}\n"
    
    return answer + sources_list, context

# 测试引用溯源
def test_citations():
    """测试引用溯源"""
    print("=== 测试引用溯源 ===")
    
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
        
        # 生成带引用的回答
        answer, context = generate_answer_with_citations(query, vectorstore, llm)
        
        print("\n上下文:")
        print(context)
        
        print("\n带引用的回答:")
        print(answer)

if __name__ == "__main__":
    test_citations()