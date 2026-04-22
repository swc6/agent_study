# BGE 与 Milvus 集成示例

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1___5"

# 连接到 Milvus 服务
def connect_to_milvus():
    print("连接到 Milvus 服务...")
    try:
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        print("连接成功！")
        return True
    except Exception as e:
        print(f"连接失败: {e}")
        print("请确保 Milvus 服务正在运行")
        return False

# 加载 BGE 模型
def load_bge_model():
    print("\n加载 BGE 模型...")
    try:
        # 尝试从本地加载
        model = SentenceTransformer(LOCAL_MODEL_PATH)
        print(f"从本地加载模型成功: {LOCAL_MODEL_PATH}")
    except Exception as e:
        print(f"本地模型加载失败: {e}")
        print("尝试从 Hugging Face 下载...")
        model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        print("从 Hugging Face 加载模型成功")
    return model

# 创建集合
def create_collection():
    print("\n创建文档向量集合...")
    
    # 定义字段
    define_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256)
    ]
    
    # 创建集合架构
    schema = CollectionSchema(fields=define_fields, description="文档向量集合")
    
    # 检查集合是否存在
    try:
        # 尝试删除已存在的集合
        collection = Collection(name="document_embeddings")
        collection.drop()
        print("删除已存在的集合")
    except:
        pass
    
    # 创建新集合
    collection = Collection(name="document_embeddings", schema=schema)
    print(f"集合 '{collection.name}' 创建成功")
    
    return collection

# 处理文档并插入数据
def process_documents(collection, model):
    print("\n处理文档并插入数据...")
    
    # 示例文档
    documents = [
        {"text": "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。", "doc_id": "doc1"},
        {"text": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。", "doc_id": "doc2"},
        {"text": "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。", "doc_id": "doc3"},
        {"text": "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。", "doc_id": "doc4"},
        {"text": "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中提取有意义的信息。", "doc_id": "doc5"}
    ]
    
    # 生成向量
    texts = [doc["text"] for doc in documents]
    print("生成文档向量...")
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype(np.float32)
    doc_ids = [doc["doc_id"] for doc in documents]
    
    # 插入数据
    data = [embeddings, texts, doc_ids]
    insert_result = collection.insert(data)
    print(f"插入了 {insert_result.insert_count} 条数据")
    print(f"插入的ID: {insert_result.primary_keys}")
    
    return documents

# 创建索引
def create_index(collection):
    print("\n创建索引...")
    
    # 定义索引参数
    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2"
    }
    
    # 创建索引
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    
    print("索引创建成功")

# 加载集合
def load_collection(collection):
    print("\n加载集合...")
    
    # 加载集合到内存
    collection.load()
    
    # 加载成功
    print("集合加载成功")

# 执行语义搜索
def semantic_search(collection, model, query):
    print(f"\n执行语义搜索: '{query}'")
    
    # 生成查询向量
    query_embedding = model.encode([query])[0]
    
    # 搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # 执行搜索
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text", "doc_id"]
    )
    
    # 打印结果
    print("搜索结果:")
    for i, hit in enumerate(results[0]):
        similarity = 1 / (1 + hit.distance)  # 转换为相似度
        print(f"{i+1}. 相似度: {similarity:.4f}, 文档ID: {hit.entity.get('doc_id')}")
        print(f"   文本: {hit.entity.get('text')}")

# 释放集合
def release_collection(collection):
    print("\n释放集合...")
    
    # 释放集合（从内存中移除）
    collection.release()
    
    # 释放成功
    print("集合释放成功")

# 主函数
def main():
    print("=== BGE 与 Milvus 集成示例 ===")
    
    # 连接 Milvus
    if not connect_to_milvus():
        return
    
    try:
        # 加载 BGE 模型
        model = load_bge_model()
        
        # 创建集合
        collection = create_collection()
        
        # 处理文档并插入数据
        documents = process_documents(collection, model)
        
        # 创建索引
        create_index(collection)
        
        # 加载集合
        load_collection(collection)
        
        # 执行搜索
        queries = [
            "什么是人工智能？",
            "机器学习和深度学习有什么关系？",
            "自然语言处理的应用场景"
        ]
        
        for query in queries:
            semantic_search(collection, model, query)
        
        # 释放集合
        release_collection(collection)
        
        print("\n操作完成！")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()