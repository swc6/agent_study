# 索引选型示例

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
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000)
    ]
    
    # 定义 schema
    schema = CollectionSchema(fields, description="文档向量集合")
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"创建集合 {collection_name} 成功")
    return collection

# 构建索引
def build_index(collection, index_type, params):
    """构建索引"""
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
            [item["content"] for item in batch]
        ]
        collection.insert(entities)
        
        # 打印进度
        processed = min(i+batch_size, total)
        print(f"插入进度: {processed}/{total}")
    
    end_time = time.time()
    print(f"插入完成，耗时: {end_time - start_time:.2f} 秒")

# 搜索数据
def search_data(collection, query_embedding, top_k=10, search_params=None):
    """搜索数据"""
    # 加载集合
    collection.load()
    
    if search_params is None:
        # 默认搜索参数
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    # 搜索
    start_time = time.time()
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["doc_id", "content"]
    )
    end_time = time.time()
    
    return results, end_time - start_time

# 生成示例数据
def generate_sample_data(num_samples=100000, dimension=768):
    """生成示例数据"""
    data = []
    for i in range(num_samples):
        item = {
            "embedding": [random.random() for _ in range(dimension)],
            "doc_id": f"doc_{i}",
            "content": f"这是文档 {i} 的内容"
        }
        data.append(item)
    return data

# 索引比较
def compare_indexes(data_size=100000):
    """比较不同索引类型的性能"""
    print("=== 索引类型比较 ===")
    
    # 连接 Milvus
    connect_milvus()
    
    # 生成示例数据
    print("生成示例数据...")
    data = generate_sample_data(num_samples=data_size)
    print(f"生成 {len(data)} 条示例数据")
    
    # 定义索引类型和参数
    index_configs = [
        {"name": "FLAT", "params": {}},  # 暴力搜索
        {"name": "IVF_FLAT", "params": {"nlist": 1024}},  # 倒排文件
        {"name": "IVF_SQ8", "params": {"nlist": 1024}},  # 量化索引
        {"name": "HNSW", "params": {"M": 16, "efConstruction": 200}}  # 层次化导航小世界图
    ]
    
    results = {}
    
    for config in index_configs:
        index_name = config["name"]
        index_params = config["params"]
        
        print(f"\n=== 测试 {index_name} 索引 ===")
        
        # 创建集合
        collection_name = f"index_test_{index_name}"
        collection = create_collection(collection_name)
        
        # 插入数据
        print("插入数据...")
        batch_insert(collection, data)
        
        # 构建索引
        print("构建索引...")
        start_time = time.time()
        build_index(collection, index_name, index_params)
        index_time = time.time() - start_time
        print(f"索引构建耗时: {index_time:.2f} 秒")
        
        # 测试搜索性能
        print("测试搜索性能...")
        query_embeddings = [[random.random() for _ in range(768)] for _ in range(50)]
        search_times = []
        
        # 根据索引类型设置搜索参数
        if index_name == "HNSW":
            search_params = {"metric_type": "L2", "params": {"ef": 100}}
        elif index_name in ["IVF_FLAT", "IVF_SQ8"]:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        else:
            search_params = {"metric_type": "L2", "params": {}}
        
        for i, query_embedding in enumerate(query_embeddings):
            _, search_time = search_data(collection, query_embedding, top_k=10, search_params=search_params)
            search_times.append(search_time)
            
            if (i + 1) % 10 == 0:
                print(f"已完成 {i+1}/50 个查询")
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"平均搜索时间: {avg_search_time:.4f} 秒")
        
        # 保存结果
        results[index_name] = {
            "index_time": index_time,
            "avg_search_time": avg_search_time
        }
    
    # 打印比较结果
    print("\n=== 索引类型比较结果 ===")
    print("索引类型 | 索引构建时间 (秒) | 平均搜索时间 (秒)")
    print("-" * 60)
    for index_name, metrics in results.items():
        print(f"{index_name:10} | {metrics['index_time']:18.2f} | {metrics['avg_search_time']:18.4f}")

# 主函数
def main():
    """主函数"""
    # 测试不同数据规模的索引性能
    data_sizes = [10000, 50000, 100000]
    
    for data_size in data_sizes:
        print(f"\n=== 测试数据规模: {data_size} ===")
        compare_indexes(data_size=data_size)

if __name__ == "__main__":
    main()