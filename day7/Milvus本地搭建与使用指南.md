# Milvus 本地搭建与使用指南

## 1. Milvus 简介

Milvus 是一个专为向量相似度搜索和分析设计的开源向量数据库，它具有高性能、可扩展和可靠的特点，是构建向量搜索系统的理想选择。

### 1.1 主要特点

- **高性能**：针对向量搜索进行了深度优化，支持毫秒级检索
- **可扩展性**：支持水平扩展，处理数十亿向量
- **多索引类型**：提供多种索引算法，适应不同场景
- **丰富的API**：支持Python、Java、Go等多种编程语言
- **与AI生态集成**：与主流深度学习框架无缝集成
- **实时更新**：支持实时添加、删除和更新向量

### 1.2 应用场景

- **语义搜索**：基于文本嵌入的智能搜索
- **图像检索**：相似图像查找
- **推荐系统**：基于用户行为的个性化推荐
- **聚类分析**：将相似向量分组
- **异常检测**：识别异常向量
- **问答系统**：找到与问题相关的文档

## 2. 环境准备

### 2.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|----------|
| CPU | 2核 | 4核以上 |
| 内存 | 8GB | 16GB以上 |
| 磁盘 | 50GB | 100GB以上 |
| 操作系统 | Linux/Windows/MacOS | Linux (Ubuntu 20.04+) |
| Docker | 19.03+ | 20.04+ |

### 2.2 安装方法

#### 2.2.1 使用 Docker Compose

这是最推荐的安装方法，简单且稳定。

**步骤1：安装 Docker 和 Docker Compose**
- 下载并安装 Docker：[https://www.docker.com/get-started](https://www.docker.com/get-started)
- 确保 Docker Compose 已安装

**步骤2：创建 docker-compose.yml 文件**

```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.2.12
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
```

**步骤3：启动 Milvus**

```bash
# 在包含 docker-compose.yml 的目录中执行
docker-compose up -d

# 检查状态
docker-compose ps
```

**步骤4：验证安装**

```bash
# 检查 Milvus 是否正常运行
docker logs milvus-standalone
```

#### 2.2.2 使用预编译包（Linux）

**步骤1：下载 Milvus**

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.2.12/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

**步骤2：启动 Milvus**

```bash
docker-compose up -d
```

## 3. Milvus 核心概念

### 3.1 数据模型

- **Collection（集合）**：向量数据的顶层容器，类似于关系数据库中的数据库
- **Partition（分区）**：Collection 的逻辑划分，用于数据管理和查询优化
- **Entity（实体）**：Collection 中的一条记录，包含主键和向量等字段
- **Field（字段）**：Entity 的属性，如主键、向量、标量字段等
- **Index（索引）**：加速向量搜索的数据结构

### 3.2 索引类型

| 索引类型 | 特点 | 适用场景 |
|---------|------|----------|
| IVF_FLAT | 基本索引，精度高，速度适中 | 中小规模数据集，对精度要求高 |
| IVF_SQ8 | 量化索引，速度快，内存占用小 | 大规模数据集，对精度要求不高 |
| IVF_PQ | 乘积量化，速度更快，内存占用更小 | 超大规模数据集，对精度要求较低 |
| HNSW | 图索引，精度最高，内存占用大 | 对搜索精度要求极高的场景 |
| ANNOY | 基于树的索引，构建速度快 | 快速原型开发，小规模数据集 |

### 3.3 距离度量

- **L2（欧氏距离）**：最常用的距离度量，适用于大多数场景
- **IP（内积）**：适用于归一化向量，计算速度快
- **Cosine（余弦相似度）**：适用于文本嵌入等方向敏感的场景

## 4. Milvus 基本操作

### 4.1 Python SDK 安装

```bash
pip install pymilvus==2.2.11
```

### 4.2 连接 Milvus

```python
from pymilvus import connections

# 连接到本地 Milvus 服务
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 检查连接状态
print(connections.has_connection("default"))
```

### 4.3 创建 Collection

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

# 定义字段
define_fields = [
    # 主键字段
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # 向量字段
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    # 标量字段（可选）
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)
]

# 创建集合架构
schema = CollectionSchema(fields=define_fields, description="测试集合")

# 创建集合
collection = Collection(name="test_collection", schema=schema)

print(f"集合 '{collection.name}' 创建成功")
print(f"集合架构: {collection.schema}")
```

### 4.4 插入数据

```python
import numpy as np

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
```

### 4.5 创建索引

```python
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
```

### 4.6 加载集合

```python
# 加载集合到内存
collection.load()

# 检查加载状态
print(f"集合是否已加载: {collection.is_loaded()}")
```

### 4.7 向量搜索

```python
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
```

### 4.8 释放集合

```python
# 释放集合（从内存中移除）
collection.release()

# 检查释放状态
print(f"集合是否已释放: {not collection.is_loaded()}")
```

### 4.9 删除 Collection

```python
# 删除集合
collection.drop()

print("集合删除成功")
```

## 5. 高级操作

### 5.1 分区管理

```python
# 创建分区
collection.create_partition(partition_name="partition_1")

# 查看分区
partitions = collection.partitions
print(f"分区列表: {[p.name for p in partitions]}")

# 向指定分区插入数据
insert_result = collection.insert(
    data=data,
    partition_name="partition_1"
)

# 按分区搜索
results = collection.search(
    data=[query_embedding[0]],
    anns_field="embedding",
    param=search_params,
    limit=3,
    partition_names=["partition_1"]
)
```

### 5.2 标量过滤

```python
# 搜索时使用标量过滤
results = collection.search(
    data=[query_embedding[0]],
    anns_field="embedding",
    param=search_params,
    limit=3,
    expr="text like '%文档%'",  # 标量过滤表达式
    output_fields=["text"]
)
```

### 5.3 批量操作

```python
# 批量插入
batch_size = 100
for i in range(0, total_data_size, batch_size):
    batch_data = data[i:i+batch_size]
    collection.insert(batch_data)

# 批量删除
ids_to_delete = [1, 2, 3]
collection.delete(expr=f"id in {ids_to_delete}")
```

### 5.4 数据备份与恢复

```python
# 备份集合
from pymilvus import utility
utility.create_backup(collection_name="test_collection", backup_name="backup_1")

# 列出备份
backups = utility.list_backups()
print(f"备份列表: {backups}")

# 恢复备份
utility.restore_backup(backup_name="backup_1", collection_name="restored_collection")
```

## 6. 性能优化

### 6.1 索引优化

- **选择合适的索引类型**：根据数据规模和精度要求选择
- **调整索引参数**：
  - IVF_FLAT：调整 nlist（通常为 sqrt(n)，n 为数据量）
  - HNSW：调整 M（通常为 16-64）和 efConstruction（通常为 100-200）

### 6.2 查询优化

- **调整搜索参数**：
  - IVF 索引：调整 nprobe（通常为 10-100）
  - HNSW 索引：调整 ef（通常为 16-128）
- **使用标量过滤**：减少需要搜索的向量数量
- **使用分区**：将数据按时间或其他维度分区
- **批量查询**：一次查询多个向量，减少网络开销

### 6.3 内存管理

- **合理设置 cache_size**：根据可用内存调整
- **及时释放不使用的集合**：使用 `release()` 方法
- **监控内存使用**：避免内存溢出

### 6.4 硬件优化

- **使用 SSD**：提高数据读写速度
- **增加内存**：Milvus 依赖内存进行索引和搜索
- **使用 GPU**：对于大规模数据，GPU 可以显著提高性能

## 7. 与 BGE 集成

### 7.1 完整流程

1. **文档处理**：使用 Docling 解析文档
2. **文本切片**：将文档分割成合适的片段
3. **向量生成**：使用 BGE 模型生成向量
4. **向量存储**：将向量存储到 Milvus
5. **语义搜索**：使用 Milvus 进行向量搜索

### 7.2 代码示例

```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 连接 Milvus
connections.connect(alias="default", host="localhost", port="19530")

# 2. 创建集合
define_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256)
]
schema = CollectionSchema(fields=define_fields, description="文档向量集合")
collection = Collection(name="document_embeddings", schema=schema)

# 3. 加载 BGE 模型
model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

# 4. 处理文档并生成向量
documents = [
    {"text": "人工智能的发展历史...", "doc_id": "doc1"},
    {"text": "机器学习的基本原理...", "doc_id": "doc2"},
    {"text": "深度学习的应用场景...", "doc_id": "doc3"}
]

# 生成向量
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype(np.float32)
doc_ids = [doc["doc_id"] for doc in documents]

# 5. 插入数据
data = [embeddings, texts, doc_ids]
insert_result = collection.insert(data)
print(f"插入了 {insert_result.insert_count} 条数据")

# 6. 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
collection.create_index(field_name="embedding", index_params=index_params)

# 7. 加载集合
collection.load()

# 8. 执行搜索
query = "人工智能的历史"
query_embedding = model.encode([query])[0]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=3,
    output_fields=["text", "doc_id"]
)

# 打印结果
print(f"查询: '{query}'")
for i, hit in enumerate(results[0]):
    similarity = 1 / (1 + hit.distance)  # 转换为相似度
    print(f"{i+1}. 相似度: {similarity:.4f}, 文档ID: {hit.entity.get('doc_id')}")
    print(f"   文本: {hit.entity.get('text')[:100]}...")

# 9. 释放集合
collection.release()
```

## 8. 常见问题与解决方案

### 8.1 连接问题

**问题**：无法连接到 Milvus 服务
**解决方案**：
- 检查 Milvus 服务是否运行：`docker-compose ps`
- 检查网络连接：确保端口 19530 可访问
- 检查防火墙设置：确保防火墙允许连接

### 8.2 内存不足

**问题**：Milvus 服务因内存不足而崩溃
**解决方案**：
- 增加系统内存
- 减少加载的集合大小
- 调整 Milvus 配置，减小 cache_size

### 8.3 索引创建失败

**问题**：索引创建失败，提示内存不足
**解决方案**：
- 使用更简单的索引类型（如 IVF_SQ8）
- 减少 nlist 参数
- 增加系统内存

### 8.4 搜索速度慢

**问题**：搜索响应时间过长
**解决方案**：
- 调整搜索参数（增加 nprobe）
- 使用更适合的索引类型
- 考虑使用 GPU 加速
- 对数据进行分区

## 9. 监控与维护

### 9.1 监控指标

- **查询延迟**：搜索响应时间
- **QPS**：每秒查询数
- **内存使用**：系统和 Milvus 内存使用情况
- **磁盘使用**：数据存储使用情况
- **索引大小**：索引占用的空间

### 9.2 日志管理

```bash
# 查看 Milvus 日志
docker logs milvus-standalone

# 查看详细日志
docker logs milvus-standalone --tail=100
```

### 9.3 定期维护

- **备份数据**：定期创建数据备份
- **优化索引**：根据数据增长情况调整索引参数
- **清理过期数据**：删除不再需要的数据
- **更新版本**：定期更新 Milvus 版本

## 10. 部署建议

### 10.1 开发环境

- **单机部署**：使用 Docker Compose 快速部署
- **资源配置**：至少 8GB 内存，4 核 CPU
- **数据量**：适合中小规模数据集（百万级向量）

### 10.2 生产环境

- **集群部署**：使用 Milvus Cluster 模式
- **资源配置**：
  - 管理节点：8GB 内存，4 核 CPU
  - 数据节点：16GB+ 内存，8 核 CPU
  - 查询节点：32GB+ 内存，16 核 CPU
- **存储**：使用 SSD 存储，预留足够空间
- **监控**：部署 Prometheus 和 Grafana 监控系统

## 11. 总结

Milvus 是一个强大的向量数据库，为构建高性能的向量搜索系统提供了坚实的基础。通过本文的学习，你应该已经掌握了：

- Milvus 的基本概念和核心功能
- 本地搭建 Milvus 服务的方法
- 创建集合、插入数据、构建索引的操作
- 执行向量搜索和优化性能的技巧
- 与 BGE 等嵌入模型的集成方法

在实际应用中，你可以根据具体需求选择合适的索引类型和参数，优化系统性能，构建满足业务需求的向量搜索系统。

随着数据量的增长和业务需求的变化，你可以考虑从单机部署升级到集群部署，以获得更好的性能和可靠性。

Milvus 作为一个活跃的开源项目，不断在改进和优化，建议关注其官方文档和社区动态，及时了解新特性和最佳实践。