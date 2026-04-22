# Milvus 基本操作示例

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

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

# 创建集合
def create_collection():
    print("\n创建集合...")
    
    # 定义字段
    define_fields = [
        # 主键字段
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 向量字段
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        # 标量字段
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
    ]
    
    # 创建集合架构
    schema = CollectionSchema(fields=define_fields, description="测试集合")
    
    # 创建集合
    collection = Collection(name="test_collection", schema=schema)
    
    print(f"集合 '{collection.name}' 创建成功")
    print(f"集合架构: {collection.schema}")
    
    return collection

# 插入数据
def insert_data(collection):
    print("\n插入数据...")
    
    # 准备数据
    texts = ["这是第一个文档", "这是第二个文档", "这是第三个文档"]
    # 生成随机向量（实际应用中使用BGE等模型生成）
    embeddings = np.random.random((3, 768)).astype(np.float32)
    
    # 插入数据
    data = [
        embeddings,
        texts
    ]
    
    # 执行插入
    insert_result = collection.insert(data)
    print(f"插入了 {insert_result.insert_count} 条数据")
    print(f"插入的ID: {insert_result.primary_keys}")
    
    return insert_result

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

# 向量搜索
def search_vectors(collection):
    print("\n执行向量搜索...")
    
    # 生成查询向量
    query_embedding = np.random.random((1, 768)).astype(np.float32)
    
    # 搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    
    # 执行搜索
    results = collection.search(
        data=[query_embedding[0]],
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["text"]
    )
    
    # 打印结果
    print("搜索结果:")
    for i, hit in enumerate(results[0]):
        print(f"{i+1}. ID: {hit.id}, 距离: {hit.distance:.4f}, 文本: {hit.entity.get('text')}")

# 释放集合
def release_collection(collection):
    print("\n释放集合...")
    
    # 释放集合（从内存中移除）
    collection.release()
    
    # 释放成功
    print("集合释放成功")

# 删除集合
def drop_collection(collection):
    print("\n删除集合...")
    
    # 删除集合
    collection.drop()
    
    print("集合删除成功")

# 主函数
def main():
    print("=== Milvus 基本操作示例 ===")
    
    # 连接 Milvus
    if not connect_to_milvus():
        return
    
    try:
        # 创建集合
        collection = create_collection()
        
        # 插入数据
        insert_data(collection)
        
        # 创建索引
        create_index(collection)
        
        # 加载集合
        load_collection(collection)
        
        # 执行搜索
        search_vectors(collection)
        
        # 释放集合
        release_collection(collection)
        
        # 删除集合
        drop_collection(collection)
        
        print("\n操作完成！")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()