# 百万级文档适配示例

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
import time
import random
import threading

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
def build_index(collection, index_type="IVF_SQ8", params=None):
    """构建索引"""
    if params is None:
        # 默认参数
        params = {"nlist": 4096}
    
    # 构建索引
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": index_type, "params": params}
    )
    print(f"构建 {index_type} 索引成功")

# 生成示例数据
def generate_sample_data(num_samples, dimension=768):
    """生成示例数据"""
    data = []
    for i in range(num_samples):
        item = {
            "embedding": [random.random() for _ in range(dimension)],
            "doc_id": f"doc_{i}",
            "content": f"这是文档 {i} 的内容",
            "metadata": {"category": f"category_{i % 100}", "created_at": "2024-01-01"}
        }
        data.append(item)
    return data

# 批量插入数据
def batch_insert(collection, data, batch_size=50000):
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

# 并行插入数据
def parallel_insert(collection, data, num_threads=4, batch_size=50000):
    """并行插入数据"""
    total = len(data)
    start_time = time.time()
    
    # 数据分片
    chunks = [data[i::num_threads] for i in range(num_threads)]
    
    # 定义插入函数
    def insert_chunk(chunk):
        batch_insert(collection, chunk, batch_size)
    
    # 创建并启动线程
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=insert_chunk, args=(chunks[i],))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    print(f"并行插入完成，耗时: {end_time - start_time:.2f} 秒")

# 缓存管理器
class CacheManager:
    """缓存管理器"""
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
        self.lock = threading.Lock()
    
    def get(self, key):
        """获取缓存"""
        with self.lock:
            if key in self.cache:
                # 更新访问顺序
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def set(self, key, value):
        """设置缓存"""
        with self.lock:
            if key in self.cache:
                # 更新访问顺序
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的项
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

# 搜索数据（带缓存）
def search_with_cache(collection, query_embedding, cache, top_k=10):
    """搜索数据（带缓存）"""
    # 生成查询向量的哈希值作为缓存键
    import hashlib
    query_hash = hashlib.md5(str(query_embedding).encode()).hexdigest()
    
    # 检查缓存
    cached_result = cache.get(query_hash)
    if cached_result:
        print("使用缓存结果")
        return cached_result
    
    # 加载集合
    collection.load()
    
    # 搜索参数
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}
    
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
    
    # 缓存结果
    cache.set(query_hash, results)
    
    return results

# 主函数
def main():
    """主函数"""
    # 连接 Milvus
    connect_milvus()
    
    # 创建集合
    collection_name = "million_documents"
    collection = create_collection(collection_name, dimension=768)
    
    # 生成示例数据（100万条）
    print("生成示例数据...")
    # 注意：生成100万条数据可能需要较长时间和大量内存
    # 这里为了演示，我们生成10万条数据
    data = generate_sample_data(num_samples=100000)
    print(f"生成 {len(data)} 条示例数据")
    
    # 并行插入数据
    print("\n并行插入数据...")
    parallel_insert(collection, data, num_threads=4, batch_size=50000)
    
    # 构建索引
    print("\n构建索引...")
    build_index(collection, index_type="IVF_SQ8", params={"nlist": 4096})
    
    # 初始化缓存
    cache = CacheManager(max_size=1000)
    
    # 测试搜索（带缓存）
    print("\n测试搜索（带缓存）...")
    query_embedding = [random.random() for _ in range(768)]
    
    # 第一次搜索（无缓存）
    print("\n第一次搜索（无缓存）:")
    results1 = search_with_cache(collection, query_embedding, cache, top_k=10)
    
    # 第二次搜索（有缓存）
    print("\n第二次搜索（有缓存）:")
    results2 = search_with_cache(collection, query_embedding, cache, top_k=10)
    
    # 打印搜索结果
    print("\n搜索结果:")
    for i, hit in enumerate(results1[0]):
        print(f"{i+1}. 文档 ID: {hit.entity.get('doc_id')}, 距离: {hit.distance:.4f}")
        print(f"   内容: {hit.entity.get('content')}")

if __name__ == "__main__":
    main()