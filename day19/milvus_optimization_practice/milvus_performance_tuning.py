# Milvus 性能调优示例

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
import time
import random

# 连接 Milvus
def connect_milvus(host="localhost", port="19530"):
    """连接 Milvus 服务器"""
    connections.connect("default", host=host, port=port)
    print("连接 Milvus 成功")

# 创建集合
def create_collection(collection_name, dimension=768):
    """创建集合"""
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    # 定义 schema
    schema = CollectionSchema(fields, description="文档向量集合")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"创建集合 {collection_name} 成功")
    return collection

# 构建索引
def build_index(collection, index_type="IVF_FLAT", params=None):
    """构建索引"""
    if params is None:
        # 默认参数
        params = {"nlist": 1024}
    
    # 构建索引
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": index_type, "params": params}
    )
    print(f"构建 {index_type} 索引成功")

# 批量插入数据
def batch_insert(collection, data, batch_size=10000):
    """批量插入数据"""
    total = len(data)
    start_time = time.time()
    
    for i in range(0, total, batch_size):
        batch = data[i:i+batch_size]
        # 准备插入数据
        entities = [
            [item["embedding"] for item in batch],
            [item["doc_id"] for item in batch],
            [item["content"] for item in batch],
            [item["metadata"] for item in batch]
        ]
        collection.insert(entities)
        
        # 打印进度
        processed = min(i+batch_size, total)
        print(f"插入进度: {processed}/{total}")
    
    end_time = time.time()
    print(f"插入完成，耗时: {end_time - start_time:.2f} 秒")

# 搜索数据
def search_data(collection, query_embedding, top_k=10, nprobe=10):
    """搜索数据"""
    # 加载集合
    collection.load()
    
    # 搜索参数
    search_params = {"metric_type": "L2", "params": {"nprobe": nprobe}}
    
    # 搜索
    start_time = time.time()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["doc_id", "content", "metadata"]
    )
    end_time = time.time()
    
    print(f"搜索完成，耗时: {end_time - start_time:.4f} 秒")
    return results

# 生成示例数据
def generate_sample_data(num_samples=100000, dimension=768):
    """生成示例数据"""
    data = []
    for i in range(num_samples):
        item = {
            "embedding": [random.random() for _ in range(dimension)],
            "doc_id": f"doc_{i}",
            "content": f"这是文档 {i} 的内容",
            "metadata": {"category": f"category_{i % 10}", "created_at": "2024-01-01"}
        }
        data.append(item)
    return data

# 性能测试
def performance_test(collection, num_queries=100, top_k=10):
    """性能测试"""
    print("\n=== 性能测试 ===")
    
    # 生成查询向量
    query_embeddings = [[random.random() for _ in range(768)] for _ in range(num_queries)]
    
    # 测试不同 nprobe 值的性能
    nprobe_values = [1, 10, 50, 100, 200]
    results = {}
    
    for nprobe in nprobe_values:
        print(f"\n测试 nprobe={nprobe}:")
        times = []
        
        for i, query_embedding in enumerate(query_embeddings):
            start_time = time.time()
            search_data(collection, query_embedding, top_k=top_k, nprobe=nprobe)
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"已完成 {i+1}/{num_queries} 个查询")
        
        avg_time = sum(times) / len(times)
        results[nprobe] = avg_time
        print(f"平均查询时间: {avg_time:.4f} 秒")
    
    # 打印结果
    print("\n=== 性能测试结果 ===")
    for nprobe, avg_time in results.items():
        print(f"nprobe={nprobe}: {avg_time:.4f} 秒")

# 主函数
def main():
    """主函数"""
    # 连接 Milvus
    connect_milvus()
    
    # 创建集合
    collection_name = "performance_test"
    collection = create_collection(collection_name, dimension=768)
    
    # 生成示例数据
    print("生成示例数据...")
    data = generate_sample_data(num_samples=100000)
    print(f"生成 {len(data)} 条示例数据")
    
    # 批量插入数据
    print("\n批量插入数据...")
    batch_insert(collection, data, batch_size=10000)
    
    # 构建索引
    print("\n构建索引...")
    build_index(collection, index_type="IVF_FLAT", params={"nlist": 1024})
    
    # 性能测试
    performance_test(collection, num_queries=50, top_k=10)

if __name__ == "__main__":
    main()