# BGE Embedding 批量处理示例
from sentence_transformers import SentenceTransformer
import numpy as np
import time
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

# 生成测试数据
def generate_test_data(num_texts=1000):
    """
    生成测试数据

    参数:
        num_texts: 文本数量

    返回:
        测试文本列表
    """
    texts = []
    for i in range(num_texts):
        texts.append(f"这是测试文本 {i}，包含一些示例内容。人工智能正在改变我们的生活，文本嵌入技术在语义搜索中发挥着重要作用。")
    return texts

# 单条处理
def process_single(model, texts):
    """
    单条处理文本

    参数:
        model: BGE模型
        texts: 文本列表

    返回:
        嵌入列表，处理时间
    """
    start_time = time.time()
    embeddings = []
    for text in texts:
        embedding = model.encode([text])[0]
        embeddings.append(embedding)
    end_time = time.time()
    return embeddings, end_time - start_time

# 批量处理
def process_batch(model, texts, batch_size=32):
    """
    批量处理文本

    参数:
        model: BGE模型
        texts: 文本列表
        batch_size: 批量大小

    返回:
        嵌入列表，处理时间
    """
    start_time = time.time()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)

    end_time = time.time()
    return embeddings, end_time - start_time

# 测试不同批量大小
def test_batch_sizes(model, texts):
    """
    测试不同批量大小的性能
    """
    batch_sizes = [1, 8, 16, 32, 64, 128]
    results = []

    for batch_size in batch_sizes:
        _, time_taken = process_batch(model, texts, batch_size)
        results.append((batch_size, time_taken))
        print(f"批量大小: {batch_size}, 处理时间: {time_taken:.2f}秒")

    return results

# 内存使用测试
def memory_usage_test(model, texts):
    """
    测试内存使用
    """
    import psutil
    import os

    # 获取初始内存使用
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 2)  # MB

    # 处理文本
    embeddings, _ = process_batch(model, texts, batch_size=32)

    # 获取处理后的内存使用
    final_memory = process.memory_info().rss / (1024 ** 2)  # MB
    memory_used = final_memory - initial_memory

    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"最终内存: {final_memory:.2f} MB")
    print(f"使用内存: {memory_used:.2f} MB")
    print(f"平均每条文本内存: {memory_used / len(texts):.4f} MB")

# 主函数
if __name__ == "__main__":
    print("=== BGE Embedding 批量处理示例 ===")

    # 加载模型
    print("加载模型...")
    model = load_model()

    # 生成测试数据
    print("生成测试数据...")
    num_texts = 1000
    texts = generate_test_data(num_texts)
    print(f"生成 {num_texts} 条测试文本")

    # 测试单条处理
    print("\n测试单条处理...")
    _, single_time = process_single(model, texts)
    print(f"单条处理时间: {single_time:.2f}秒")

    # 测试批量处理
    print("\n测试批量处理...")
    _, batch_time = process_batch(model, texts, batch_size=32)
    print(f"批量处理时间: {batch_time:.2f}秒")
    print(f"加速比: {single_time / batch_time:.2f}x")

    # 测试不同批量大小
    print("\n测试不同批量大小...")
    test_batch_sizes(model, texts)

    # 测试内存使用
    print("\n测试内存使用...")
    memory_usage_test(model, texts)

    print("\n所有测试完成!")