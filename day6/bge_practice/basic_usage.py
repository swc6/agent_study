# BGE Embedding 基本使用示例
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1___5"

# 加载BGE模型
def load_model(model_path=None):
    """
    加载BGE模型

    参数:
        model_path: 本地模型路径，如果为None则从Hugging Face下载

    返回:
        加载好的模型
    """
    if model_path is None:
        model_path = LOCAL_MODEL_PATH

    # 检查本地路径是否存在
    if os.path.exists(model_path):
        print(f"从本地加载模型: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"本地模型不存在，从Hugging Face下载...")
        model_name = 'BAAI/bge-base-zh-v1.5'
        print(f"加载模型: {model_name}")
        model = SentenceTransformer(model_name)

    print("模型加载完成!")
    return model

# 基本嵌入示例
def basic_embedding(model):
    """
    基本嵌入示例
    """
    print("\n=== 基本嵌入示例 ===")

    # 测试文本
    texts = [
        "BGE Embedding 是一个强大的文本嵌入模型",
        "人工智能正在快速发展",
        "文本嵌入在语义搜索中非常重要",
        "Python 是一种流行的编程语言"
    ]

    # 生成嵌入
    print("生成嵌入...")
    embeddings = model.encode(texts)

    # 打印结果
    print(f"文本数量: {len(texts)}")
    print(f"嵌入维度: {embeddings[0].shape}")
    print("\n嵌入向量示例:")
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        print(f"\n文本 {i+1}: {text}")
        print(f"嵌入前5维: {embedding[:5].round(4)}")
        print(f"嵌入范数: {np.linalg.norm(embedding):.4f}")

# 相似度计算示例
def similarity_example(model):
    """
    相似度计算示例
    """
    print("\n=== 相似度计算示例 ===")

    # 测试文本对
    text_pairs = [
        ("猫是一种动物", "狗也是一种动物"),  # 相似
        ("猫是一种动物", "汽车是一种交通工具"),  # 不相似
        ("人工智能发展迅速", "AI技术进步很快"),  # 相似
        ("天气很好", "今天下雨了")  # 不相似
    ]

    # 计算相似度
    for text1, text2 in text_pairs:
        emb1 = model.encode([text1])[0]
        emb2 = model.encode([text2])[0]
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        print(f"'{text1}' 与 '{text2}' 的相似度: {similarity:.4f}")

# 语义搜索示例
def semantic_search_example(model):
    """
    语义搜索示例
    """
    print("\n=== 语义搜索示例 ===")

    # 文档库
    documents = [
        "人工智能的发展历史可以追溯到1950年代",
        "机器学习是人工智能的一个重要分支",
        "深度学习在计算机视觉领域取得了重大突破",
        "自然语言处理是人工智能的重要应用领域",
        "机器人技术是人工智能的另一个重要应用"
    ]

    # 生成文档嵌入
    doc_embeddings = model.encode(documents)

    # 测试查询
    queries = [
        "人工智能的历史",
        "机器学习是什么",
        "计算机视觉的进展",
        "自然语言处理应用",
        "机器人技术"
    ]

    # 执行搜索
    for query in queries:
        # 生成查询嵌入
        query_embedding = model.encode([query])[0]

        # 计算相似度
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # 获取最相似的文档
        most_similar_idx = np.argmax(similarities)

        print(f"\n查询: '{query}'")
        print(f"最相关文档: '{documents[most_similar_idx]}'")
        print(f"相似度: {similarities[most_similar_idx]:.4f}")

# 主函数
if __name__ == "__main__":
    # 加载模型
    model = load_model()

    # 运行示例
    basic_embedding(model)
    similarity_example(model)
    semantic_search_example(model)

    print("\n所有示例运行完成!")