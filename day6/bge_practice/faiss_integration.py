# BGE Embedding 与 FAISS 集成示例
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1___5"

# 加载模型
def load_model():
    """
    加载BGE模型
    """
    # 检查本地路径是否存在
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"从本地加载模型: {LOCAL_MODEL_PATH}")
        model = SentenceTransformer(LOCAL_MODEL_PATH)
    else:
        print(f"本地模型不存在，从Hugging Face下载...")
        model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    return model

# 创建FAISS索引
def create_faiss_index(embeddings):
    """
    创建FAISS索引

    参数:
        embeddings: 嵌入向量列表

    返回:
        FAISS索引
    """
    # 获取嵌入维度
    dimension = embeddings.shape[1]

    # 创建索引 (L2距离)
    index = faiss.IndexFlatL2(dimension)

    # 添加向量
    index.add(embeddings)

    print(f"创建FAISS索引完成，包含 {index.ntotal} 个向量")
    return index

# 语义搜索
def semantic_search(index, model, query, documents, k=3):
    """
    语义搜索

    参数:
        index: FAISS索引
        model: BGE模型
        query: 查询文本
        documents: 文档列表
        k: 返回结果数量

    返回:
        搜索结果 [(文档, 距离)]
    """
    # 生成查询嵌入
    query_embedding = model.encode([query])

    # 搜索
    distances, indices = index.search(query_embedding, k)

    # 整理结果
    results = []
    for i in range(k):
        if indices[0][i] < len(documents):
            results.append((documents[indices[0][i]], distances[0][i]))

    return results

# 主函数
if __name__ == "__main__":
    print("=== BGE Embedding 与 FAISS 集成示例 ===")

    # 加载模型
    print("加载模型...")
    model = load_model()

    # 准备文档
    documents = [
        "人工智能的发展历史可以追溯到1950年代，当时图灵提出了著名的图灵测试。",
        "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习。",
        "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。",
        "计算机视觉是人工智能的一个应用领域，专注于让计算机理解和解释图像。",
        "自然语言处理是人工智能的另一个重要领域，致力于让计算机理解和生成人类语言。",
        "机器人技术结合了人工智能、机械工程和电子工程，创造出能够执行任务的机器。",
        "强化学习是一种机器学习方法，通过试错来学习最优策略。",
        "知识图谱是一种结构化的知识表示方法，用于存储和检索实体之间的关系。",
        "计算机视觉在自动驾驶、安防监控和医疗影像分析等领域有广泛应用。",
        "自然语言处理技术已经在机器翻译、语音识别和情感分析等任务中取得了重大突破。"
    ]

    print(f"准备了 {len(documents)} 个文档")

    # 生成文档嵌入
    print("生成文档嵌入...")
    doc_embeddings = model.encode(documents)
    doc_embeddings = np.array(doc_embeddings)

    # 创建FAISS索引
    print("创建FAISS索引...")
    index = create_faiss_index(doc_embeddings)

    # 测试查询
    queries = [
        "人工智能的历史",
        "机器学习的方法",
        "计算机视觉的应用",
        "自然语言处理技术",
        "机器人技术的发展"
    ]

    print("\n测试语义搜索...")
    for query in queries:
        print(f"\n查询: '{query}'")
        results = semantic_search(index, model, query, documents, k=3)

        for i, (doc, distance) in enumerate(results):
            # 计算相似度（距离的倒数）
            similarity = 1 / (1 + distance)
            print(f"  {i+1}. 相似度: {similarity:.4f}, 文档: {doc[:100]}...")

    # 测试动态添加文档
    print("\n测试动态添加文档...")
    new_documents = [
        "生成式AI是人工智能的一个新兴领域，能够创建新的内容，如文本、图像和音频。",
        "大语言模型是生成式AI的重要应用，如GPT、BERT等模型。"
    ]

    # 生成新文档的嵌入
    new_embeddings = model.encode(new_documents)
    new_embeddings = np.array(new_embeddings)

    # 添加到索引
    index.add(new_embeddings)
    print(f"添加了 {len(new_documents)} 个新文档，索引总文档数: {index.ntotal}")

    # 更新文档列表
    documents.extend(new_documents)

    # 测试新查询
    print("\n测试新查询...")
    new_query = "生成式AI和大语言模型"
    results = semantic_search(index, model, new_query, documents, k=3)

    print(f"查询: '{new_query}'")
    for i, (doc, distance) in enumerate(results):
        similarity = 1 / (1 + distance)
        print(f"  {i+1}. 相似度: {similarity:.4f}, 文档: {doc[:100]}...")

    print("\n所有测试完成!")