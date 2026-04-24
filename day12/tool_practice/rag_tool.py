# RAG工具封装示例

from langchain.tools import BaseTool
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.dashscope import DashScope
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class RAGInput(BaseModel):
    """RAG工具的输入参数"""
    query: str = Field(description="查询文本")
    k: int = Field(default=3, description="返回的文档数量")

class RAGTool(BaseTool):
    """RAG工具，用于从知识库中检索信息"""
    name: str = "rag"
    description: str = "从知识库中检索相关信息并生成答案"
    args_schema: type[BaseModel] = RAGInput
    
    def __init__(self, vectorstore):
        """
        初始化RAG工具
        
        参数:
            vectorstore: 向量存储对象
        """
        super().__init__()
        self.vectorstore = vectorstore
        # 初始化LLM
        self.llm = DashScope(
            model="qwen-plus",
            temperature=0.7
        )
        # 创建检索问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
    
    def _run(self, query: str, k: int = 3) -> str:
        """从知识库中检索信息并生成答案"""
        try:
            # 执行检索问答
            result = self.qa_chain.invoke({
                "query": query
            })
            
            # 整理结果
            answer = result["result"]
            sources = [doc.metadata.get("source", "未知") for doc in result.get("source_documents", [])]
            
            final_result = f"答案: {answer}\n\n来源: {', '.join(sources)}"
            return final_result
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    async def _arun(self, query: str, k: int = 3) -> str:
        """异步从知识库中检索信息并生成答案"""
        return self._run(query, k)

class SimpleRAGTool(BaseTool):
    """简单RAG工具，只返回检索到的文档"""
    name: str = "simple_rag"
    description: str = "从知识库中检索相关文档"
    args_schema: type[BaseModel] = RAGInput
    
    def __init__(self, vectorstore):
        """
        初始化简单RAG工具
        
        参数:
            vectorstore: 向量存储对象
        """
        super().__init__()
        self.vectorstore = vectorstore
    
    def _run(self, query: str, k: int = 3) -> str:
        """从知识库中检索文档"""
        try:
            # 检索相关文档
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # 整理检索结果
            result = "检索到的相关信息：\n"
            for i, doc in enumerate(docs):
                result += f"\n{i+1}. {doc.page_content[:200]}..."
                if doc.metadata:
                    result += f" (来源: {doc.metadata.get('source', '未知')})"
            
            return result
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    async def _arun(self, query: str, k: int = 3) -> str:
        """异步从知识库中检索文档"""
        return self._run(query, k)

# 示例用法
def create_sample_vectorstore():
    """创建示例向量存储"""
    # 示例文档
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。"
    ]
    
    # 加载BGE模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5"
    )
    
    # 创建向量存储
    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embeddings,
        metadatas=[{"source": f"doc_{i+1}"} for i in range(len(documents))]
    )
    
    return vectorstore

def test_rag_tools():
    print("=== 测试RAG工具 ===")
    
    # 创建示例向量存储
    vectorstore = create_sample_vectorstore()
    print("向量存储创建成功！")
    
    # 测试简单RAG工具
    simple_rag_tool = SimpleRAGTool(vectorstore)
    print("\n测试简单RAG工具:")
    result = simple_rag_tool._run("什么是人工智能？")
    print(result)
    
    # 测试高级RAG工具
    rag_tool = RAGTool(vectorstore)
    print("\n测试高级RAG工具:")
    result = rag_tool._run("什么是机器学习？")
    print(result)

if __name__ == "__main__":
    test_rag_tools()