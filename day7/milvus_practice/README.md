# Milvus 练习项目

本目录包含Milvus向量数据库的使用示例和实践代码，帮助你掌握向量数据库的应用。

## 环境准备

### 安装依赖

```bash
# 进入milvus_practice目录
cd day7\milvus_practice

# 安装依赖
pip install -r requirements.txt
```

### 启动 Milvus 服务

在运行示例代码前，需要启动 Milvus 服务。推荐使用 Docker Compose 方式部署：

1. **创建 docker-compose.yml 文件**

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

2. **启动服务**

```bash
# 在包含 docker-compose.yml 的目录中执行
docker-compose up -d

# 检查状态
docker-compose ps
```

3. **验证服务**

```bash
# 查看 Milvus 日志
docker logs milvus-standalone
```

## 示例代码

### 1. 基本操作 (`basic_operations.py`)

演示 Milvus 的基本功能：
- 连接 Milvus 服务
- 创建集合
- 插入数据
- 创建索引
- 执行向量搜索
- 释放和删除集合

**运行方法**：
```bash
python basic_operations.py
```

### 2. 与 BGE 集成 (`bge_integration.py`)

演示如何将 BGE 嵌入模型与 Milvus 集成：
- 加载 BGE 模型（优先使用本地模型）
- 生成文档向量
- 将向量存储到 Milvus
- 执行语义搜索

**运行方法**：
```bash
python bge_integration.py
```

## 学习路径

1. **基础篇**：运行 `basic_operations.py`，了解 Milvus 的基本操作
2. **集成篇**：运行 `bge_integration.py`，学习如何与 BGE 模型集成
3. **实践篇**：尝试构建自己的向量搜索系统

## 学习建议

1. **理解核心概念**：
   - Collection（集合）：向量数据的容器
   - Field（字段）：数据的属性
   - Index（索引）：加速搜索的数据结构
   - Entity（实体）：一条向量记录

2. **索引选择**：
   - 小规模数据：IVF_FLAT（精度高）
   - 大规模数据：IVF_SQ8（速度快）
   - 对精度要求高：HNSW（精度最高）

3. **参数调优**：
   - IVF 索引：调整 nlist（通常为 sqrt(n)，n 为数据量）
   - 搜索参数：调整 nprobe（通常为 10-100）

4. **性能优化**：
   - 使用批量插入减少网络开销
   - 合理设置缓存大小
   - 考虑使用 GPU 加速

5. **实际应用**：
   - 构建语义搜索引擎
   - 实现相似图像检索
   - 开发推荐系统

## 常见问题

### 1. 连接失败
- 检查 Milvus 服务是否运行
- 确认端口 19530 是否可访问
- 检查防火墙设置

### 2. 内存不足
- 减少加载的集合大小
- 调整 Milvus 配置
- 增加系统内存

### 3. 索引创建失败
- 使用更简单的索引类型
- 减少 nlist 参数
- 增加系统内存

### 4. 搜索速度慢
- 调整搜索参数（增加 nprobe）
- 使用更适合的索引类型
- 考虑使用 GPU 加速

## 扩展实验

1. **不同索引类型对比**：测试 IVF_FLAT、IVF_SQ8 和 HNSW 的性能差异
2. **大规模数据测试**：插入和搜索百万级向量
3. **混合检索**：结合标量过滤和向量搜索
4. **实时更新**：测试动态添加和删除数据
5. **分布式部署**：尝试 Milvus Cluster 模式

## 参考资源

- [Milvus 官方文档](https://milvus.io/docs)
- [Milvus GitHub 仓库](https://github.com/milvus-io/milvus)
- [pymilvus 文档](https://milvus.io/docs/sdk/python.md)
- [BGE 官方文档](https://github.com/FlagOpen/FlagEmbedding)

## 注意事项

- Milvus 服务需要占用较多内存（建议至少 8GB）
- 首次运行时，BGE 模型会自动下载（如果本地模型不存在）
- 生产环境中，建议使用集群部署以获得更好的性能和可靠性

通过这些示例和实验，你将掌握 Milvus 向量数据库的使用方法，为构建高性能的向量搜索系统打下基础。