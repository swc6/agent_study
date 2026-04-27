# 批量向量构建示例

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import os

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")

# 加载文档
def load_document(path):
    """加载文档"""
    try:
        loader = TextLoader(path, encoding='utf-8')
        documents = loader.load()
        return documents[0]
    except Exception as e:
        print(f"加载文档 {path} 失败: {str(e)}")
        return None

# 分块文档
def chunk_document(document, chunk_size=1000, chunk_overlap=200):
    """分块文档"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents([document])
    return chunks

# 批量生成向量
def batch_generate_embeddings(texts, embedding_model, batch_size=32):
    """批量生成向量"""
    embeddings = []
    errors = []
    start_time = time.time()
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            # 批量生成向量
            batch_embeddings = embedding_model.encode(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            # 单个批次失败，尝试逐个处理
            print(f"批量处理失败，尝试逐个处理: {str(e)}")
            for j, text in enumerate(batch):
                try:
                    embedding = embedding_model.encode([text])[0]
                    embeddings.append(embedding)
                except Exception as e2:
                    errors.append((i+j, str(e2)))
        
        # 打印进度
        processed = min(i + batch_size, len(texts))
        print(f"生成向量进度: {processed}/{len(texts)}")
    
    end_time = time.time()
    print(f"向量生成完成，耗时: {end_time - start_time:.2f} 秒")
    
    return embeddings, errors

# 批量构建向量索引
def build_vector_index(chunks, embeddings_model):
    """批量构建向量索引"""
    print(f"开始构建向量索引，共 {len(chunks)} 个文档块")
    start_time = time.time()
    
    # 构建FAISS索引
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    
    end_time = time.time()
    print(f"向量索引构建完成，耗时: {end_time - start_time:.2f} 秒")
    return vectorstore

# 测试批量向量构建
def test_batch_vector_building():
    """测试批量向量构建"""
    print("=== 测试批量向量构建 ===")
    
    # 创建示例文档
    sample_docs = [
        "# 文档1\n人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "# 文档2\n机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "# 文档3\n深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "# 文档4\n自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "# 文档5\n计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。"
    ]
    
    # 保存示例文档
    doc_paths = []
    for i, content in enumerate(sample_docs):
        path = f"sample_doc_{i+1}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        doc_paths.append(path)
    
    # 加载和分块文档
    print("\n=== 加载和分块文档 ===")
    all_chunks = []
    for path in doc_paths:
        doc = load_document(path)
        if doc:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
    
    print(f"共处理 {len(all_chunks)} 个文档块")
    
    # 测试批量向量生成
    print("\n=== 测试批量向量生成 ===")
    texts = [chunk.page_content for chunk in all_chunks]
    embeddings_list, errors = batch_generate_embeddings(texts, embeddings, batch_size=4)
    
    print(f"向量生成完成，成功 {len(embeddings_list)} 个，失败 {len(errors)} 个")
    
    # 测试向量索引构建
    print("\n=== 测试向量索引构建 ===")
    vectorstore = build_vector_index(all_chunks, embeddings)
    
    # 保存索引
    vectorstore.save_local("vectorstore")
    print("向量索引已保存")
    
    # 测试检索
    print("\n=== 测试检索 ===")
    queries = ["什么是人工智能", "机器学习的定义", "深度学习的应用"]
    for query in queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\n查询: '{query}'")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.page_content[:100]}...")
    
    # 清理临时文件
    for path in doc_paths:
        if os.path.exists(path):
            os.remove(path)
    
    # 清理向量索引
    if os.path.exists("vectorstore"):
        import shutil
        shutil.rmtree("vectorstore")

if __name__ == "__main__":
    test_batch_vector_building()