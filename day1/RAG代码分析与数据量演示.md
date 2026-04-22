# RAG代码分析与数据量演示

## 1. RAG架构概述

RAG（Retrieval-Augmented Generation）是一种结合了信息检索和生成式AI的技术架构，通过在生成回答前检索相关信息，显著提升了大语言模型的知识准确性和时效性。

### 核心组件

1. **文档处理**：解析和处理原始文档
2. **文本切片**：将文档分割成有意义的文本片段
3. **向量嵌入**：将文本转换为向量表示
4. **向量存储**：存储和管理向量数据
5. **检索系统**：根据查询检索相关文档
6. **重排序**：优化检索结果的排序
7. **生成模型**：基于检索结果生成回答

## 2. 完整RAG代码实现

### 2.1 环境搭建

```python
# 安装依赖
!pip install langchain langchain-community sentence-transformers pymilvus pymilvus[model] rank_bm25
```

### 2.2 完整RAG实现

```python
import os
import textwrap
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 1. 文档加载与处理
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents

# 2. 文本切片
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# 3. 向量嵌入与存储
def create_vector_store(chunks):
    # 使用BGE Embedding模型
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 创建Milvus向量存储
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={
            "uri": "http://localhost:19530"
        }
    )
    return vector_store

# 4. 混合检索系统
def create_retriever(vector_store, chunks):
    # 向量检索
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # BM25检索
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    
    # 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever

# 5. 构建RAG链
def build_rag_chain(retriever):
    # 提示词模板
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    
    {context}
    
    Question: {question}
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 使用OpenAI模型
    llm = OpenAI(temperature=0.1)
    
    # 构建RAG链
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return rag_chain

# 6. 主函数
def main():
    # 加载文档
    documents = load_documents("documents")
    print(f"加载了 {len(documents)} 个文档")
    
    # 文本切片
    chunks = split_documents(documents)
    print(f"生成了 {len(chunks)} 个文本切片")
    
    # 创建向量存储
    vector_store = create_vector_store(chunks)
    print("向量存储创建完成")
    
    # 创建检索器
    retriever = create_retriever(vector_store, chunks)
    print("检索器创建完成")
    
    # 构建RAG链
    rag_chain = build_rag_chain(retriever)
    print("RAG链构建完成")
    
    # 测试查询
    test_questions = [
        "What is RAG?",
        "How does vector embedding work?",
        "What are the advantages of hybrid retrieval?"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = rag_chain.invoke({"query": question})
        print(f"回答: {result['result']}")

if __name__ == "__main__":
    main()
```

## 3. 数据量演示与性能分析

### 3.1 不同数据量下的性能测试

```python
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 性能测试函数
def performance_test(document_count):
    results = []
    
    for count in document_count:
        print(f"测试 {count} 个文档...")
        
        # 加载指定数量的文档
        documents = []
        for i in range(min(count, len(os.listdir("documents")))):
            filename = os.listdir("documents")[i]
            if filename.endswith('.pdf'):
                filepath = os.path.join("documents", filename)
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
        
        # 1. 文本切片时间
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        split_time = time.time() - start_time
        
        # 2. 向量存储创建时间
        start_time = time.time()
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
        vector_store = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_args={"uri": "http://localhost:19530"}
        )
        store_time = time.time() - start_time
        
        # 3. 检索时间测试
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        
        # 测试检索时间
        test_query = "What is RAG?"
        start_time = time.time()
        retrieved_docs = ensemble_retriever.get_relevant_documents(test_query)
        retrieval_time = time.time() - start_time
        
        # 4. 生成时间测试
        llm = OpenAI(temperature=0.1)
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        start_time = time.time()
        result = rag_chain.invoke({"query": test_query})
        generation_time = time.time() - start_time
        
        # 记录结果
        results.append({
            "document_count": count,
            "chunk_count": len(chunks),
            "split_time": split_time,
            "store_time": store_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": split_time + store_time + retrieval_time + generation_time
        })
        
        print(f"完成 {count} 个文档的测试")
    
    return pd.DataFrame(results)

# 运行性能测试
document_counts = [1, 5, 10, 20, 50]
df_results = performance_test(document_counts)

# 保存结果
df_results.to_csv("rag_performance_test.csv", index=False)

# 显示结果
print("\n性能测试结果:")
print(df_results)

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df_results['document_count'], df_results['split_time'], label='文本切片')
plt.plot(df_results['document_count'], df_results['store_time'], label='向量存储')
plt.plot(df_results['document_count'], df_results['retrieval_time'], label='检索')
plt.plot(df_results['document_count'], df_results['generation_time'], label='生成')
plt.plot(df_results['document_count'], df_results['total_time'], label='总时间')
plt.xlabel('文档数量')
plt.ylabel('时间 (秒)')
plt.title('不同数据量下的RAG性能')
plt.legend()
plt.grid(True)
plt.savefig('rag_performance.png')
plt.show()
```

### 3.2 测试结果分析

#### 预期结果示例

| 文档数量 | 切片数量 | 切片时间 | 存储时间 | 检索时间 | 生成时间 | 总时间 |
|---------|---------|---------|---------|---------|---------|--------|
| 1       | 50      | 0.1     | 2.5     | 0.3     | 1.2     | 4.1    |
| 5       | 250     | 0.5     | 10.2    | 0.4     | 1.3     | 12.4   |
| 10      | 500     | 1.0     | 22.5    | 0.5     | 1.4     | 25.4   |
| 20      | 1000    | 2.0     | 45.0    | 0.6     | 1.5     | 49.1   |
| 50      | 2500    | 5.0     | 110.0   | 0.8     | 1.6     | 117.4  |

#### 性能分析

1. **文本切片**：时间与文档数量近似线性增长
2. **向量存储**：时间与文档数量近似线性增长，是最耗时的环节
3. **检索时间**：随着文档数量增加，检索时间略有增加但增长缓慢
4. **生成时间**：几乎不受文档数量影响，主要取决于LLM的响应速度

### 3.3 大规模数据优化策略

1. **批量处理**：
   ```python
   def batch_process(documents, batch_size=100):
       """批量处理文档"""
       all_chunks = []
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i+batch_size]
           chunks = split_documents(batch)
           all_chunks.extend(chunks)
       return all_chunks
   ```

2. **异步处理**：
   ```python
   import asyncio
   
   async def async_embed(chunks):
       """异步嵌入处理"""
       # 实现异步嵌入逻辑
       pass
   ```

3. **索引优化**：
   ```python
   # Milvus索引优化
   vector_store = Milvus.from_documents(
       documents=chunks,
       embedding=embeddings,
       connection_args={"uri": "http://localhost:19530"},
       index_params={
           "index_type": "HNSW",
           "metric_type": "L2",
           "params": {"M": 16, "efConstruction": 200}
       }
   )
   ```

## 4. 代码优化建议

### 4.1 性能优化

1. **使用GPU加速**：
   ```python
   embeddings = HuggingFaceBgeEmbeddings(
       model_name="BAAI/bge-large-en-v1.5",
       model_kwargs={"device": "cuda"}  # 使用GPU
   )
   ```

2. **缓存机制**：
   ```python
   import joblib
   
   # 缓存嵌入结果
   if os.path.exists("embeddings_cache.joblib"):
       embeddings_cache = joblib.load("embeddings_cache.joblib")
   else:
       embeddings_cache = {}
       joblib.dump(embeddings_cache, "embeddings_cache.joblib")
   ```

3. **并行处理**：
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   def process_documents_parallel(documents):
       with ThreadPoolExecutor(max_workers=4) as executor:
           chunks = list(executor.map(process_single_document, documents))
       return chunks
   ```

### 4.2 功能优化

1. **多模态支持**：
   ```python
   from langchain_community.document_loaders import ImageLoader
   
   # 处理图像文档
   image_loader = ImageLoader("images/")
   image_documents = image_loader.load()
   ```

2. **增量更新**：
   ```python
   def update_vector_store(vector_store, new_documents):
       """增量更新向量存储"""
       new_chunks = split_documents(new_documents)
       vector_store.add_documents(new_chunks)
       return vector_store
   ```

3. **自定义检索策略**：
   ```python
   def custom_retrieval_strategy(query, vector_store, top_k=5):
       """自定义检索策略"""
       # 实现更复杂的检索逻辑
       pass
   ```

## 5. 实际应用示例

### 5.1 知识库问答系统

```python
class KnowledgeBaseQA:
    def __init__(self, document_directory):
        self.documents = load_documents(document_directory)
        self.chunks = split_documents(self.documents)
        self.vector_store = create_vector_store(self.chunks)
        self.retriever = create_retriever(self.vector_store, self.chunks)
        self.rag_chain = build_rag_chain(self.retriever)
    
    def ask(self, question):
        """回答问题"""
        result = self.rag_chain.invoke({"query": question})
        return result['result']
    
    def add_document(self, document_path):
        """添加新文档"""
        loader = PyPDFLoader(document_path)
        new_documents = loader.load()
        new_chunks = split_documents(new_documents)
        self.vector_store.add_documents(new_chunks)
        self.chunks.extend(new_chunks)
        self.retriever = create_retriever(self.vector_store, self.chunks)
        self.rag_chain = build_rag_chain(self.retriever)

# 使用示例
kb_qa = KnowledgeBaseQA("documents")

# 提问
answer = kb_qa.ask("What is the difference between RAG and fine-tuning?")
print(answer)

# 添加新文档
kb_qa.add_document("new_document.pdf")

# 再次提问
answer = kb_qa.ask("What's new in the latest document?")
print(answer)
```

### 5.2 企业文档管理系统

```python
class EnterpriseDocumentSystem:
    def __init__(self):
        self.knowledge_bases = {}
    
    def create_knowledge_base(self, name, document_directory):
        """创建知识库"""
        self.knowledge_bases[name] = KnowledgeBaseQA(document_directory)
        return f"知识库 {name} 创建成功"
    
    def ask_knowledge_base(self, name, question):
        """向知识库提问"""
        if name in self.knowledge_bases:
            return self.knowledge_bases[name].ask(question)
        else:
            return f"知识库 {name} 不存在"
    
    def add_document_to_knowledge_base(self, name, document_path):
        """向知识库添加文档"""
        if name in self.knowledge_bases:
            self.knowledge_bases[name].add_document(document_path)
            return f"文档添加到知识库 {name} 成功"
        else:
            return f"知识库 {name} 不存在"

# 使用示例
doc_system = EnterpriseDocumentSystem()

# 创建知识库
doc_system.create_knowledge_base("HR", "hr_documents")
doc_system.create_knowledge_base("Finance", "finance_documents")

# 向知识库提问
hr_answer = doc_system.ask_knowledge_base("HR", "What are the company's vacation policies?")
finance_answer = doc_system.ask_knowledge_base("Finance", "What is the expense reimbursement process?")

print("HR回答:", hr_answer)
print("Finance回答:", finance_answer)
```

## 6. 总结

RAG技术通过结合检索和生成，显著提升了大语言模型的知识能力。本文提供了完整的RAG代码实现，并通过性能测试分析了不同数据量下的系统表现。

### 关键发现

1. **性能瓶颈**：向量存储是最耗时的环节，特别是在处理大量文档时
2. **可扩展性**：检索时间增长缓慢，说明RAG系统具有良好的可扩展性
3. **优化方向**：
   - 使用GPU加速向量嵌入
   - 实现批量和异步处理
   - 优化向量数据库索引
   - 采用增量更新策略

### 未来发展

RAG技术正在快速发展，未来的方向包括：

1. **多模态RAG**：整合文本、图像、视频等多种数据类型
2. **自适应检索**：根据查询自动调整检索策略
3. **知识图谱增强**：结合知识图谱提升检索质量
4. **联邦RAG**：在保护隐私的前提下实现跨数据源检索

通过本文的代码分析和性能演示，你应该对RAG系统的工作原理和性能特点有了更深入的理解，为构建自己的RAG应用打下了坚实的基础。