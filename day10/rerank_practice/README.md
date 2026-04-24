# BGE Rerank 重排序练习项目

本目录包含BGE Rerank重排序的使用示例和实践代码，帮助你掌握重排序技术的应用。

## 环境准备

### 安装依赖

```bash
# 进入rerank_practice目录
cd day10\rerank_practice

# 安装依赖
pip install -r requirements.txt
```

### 硬件要求

- **基本要求**：4GB+ RAM（使用bge-reranker-base模型）
- **推荐配置**：8GB+ RAM，支持GPU加速

## 示例代码

### 1. 基本重排序 (`basic_rerank.py`)

演示BGE Rerank的基本功能：
- 加载模型（支持本地模型）
- 添加文档
- 执行重排序
- 批量处理

**运行方法**：
```bash
python basic_rerank.py
```

### 2. 与混合检索集成 (`hybrid_rerank_integration.py`)

展示如何将BGE Rerank与混合检索集成：
- 实现纯向量检索
- 实现纯TF-IDF检索
- 实现混合检索
- 实现混合检索 + 重排序
- 比较不同方法的效果

**运行方法**：
```bash
python hybrid_rerank_integration.py
```

## 学习路径

1. **基础篇**：运行`basic_rerank.py`，了解BGE Rerank的基本功能
2. **集成篇**：运行`hybrid_rerank_integration.py`，学习如何与混合检索集成
3. **比较篇**：分析不同检索方法的效果差异
4. **应用篇**：将重排序应用到实际项目中

## 模型选择

BGE Rerank提供了多个模型版本，你可以根据硬件情况选择：

| 模型名称 | 参数量 | 适用场景 | 硬件要求 |
|---------|-------|---------|---------|
| bge-reranker-base | 1.2B | 一般场景 | 4GB+ RAM |
| bge-reranker-large | 3.3B | 高精度场景 | 8GB+ RAM |
| bge-reranker-v2-m3 | 1.3B | 多语言场景 | 4GB+ RAM |

要使用不同模型，只需修改代码中的模型名称：

```python
# 例如使用large模型
model = CrossEncoder('BAAI/bge-reranker-large')
```

## 首次运行注意事项

### 模型下载
- 首次运行时会自动从Hugging Face下载模型
- 下载时间取决于网络连接速度
- 如果下载失败，请检查网络连接或尝试使用代理

### 常见问题
1. **下载速度慢**：可以使用镜像站点或设置代理
2. **内存不足**：尝试使用更小的模型（如bge-reranker-base）
3. **模型加载失败**：检查Hugging Face访问是否正常

## 实际应用场景

BGE Rerank可以应用于以下场景：

1. **搜索引擎**：提高搜索结果的相关性
2. **问答系统**：找到最相关的文档用于生成答案
3. **推荐系统**：优化推荐内容的排序
4. **信息检索**：在大型文档库中找到最相关的信息
5. **学术搜索**：提高学术论文的检索质量

## 最佳实践

1. **候选数量选择**：
   - 简单查询：5-10个候选
   - 复杂查询：10-20个候选
   - 专业领域：20-30个候选

2. **性能优化**：
   - 使用批量处理提高效率
   - 考虑模型量化减少内存使用
   - 缓存常见查询的重排序结果

3. **集成策略**：
   - 初步检索使用快速方法（如BM25 + 向量）
   - 重排序使用高精度模型
   - 根据查询复杂度动态调整策略

4. **评估方法**：
   - 使用NDCG@k、MRR等指标评估重排序效果
   - 进行A/B测试，比较有无重排序的差异
   - 分析不同类型查询的重排序效果

## 扩展实验

尝试以下实验来深入学习：

1. **不同模型对比**：比较bge-reranker-base和bge-reranker-large的性能和效果
2. **候选数量优化**：测试不同候选数量对重排序效果的影响
3. **批量大小调整**：找到最佳的批量处理大小
4. **与Milvus集成**：将重排序与Milvus向量数据库集成
5. **实时应用**：构建一个简单的Web服务提供重排序功能

## 参考资源

- [BGE官方文档](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence-Transformers文档](https://www.sbert.net/)
- [Cross-Encoder文档](https://www.sbert.net/docs/pretrained_cross-encoders.html)
- [重排序技术综述](https://arxiv.org/abs/2010.00485)

## 依赖说明

- `sentence-transformers`: 提供BGE Rerank模型的接口
- `numpy`: 数组操作
- `scikit-learn`: 相似度计算
- `jieba`: 中文分词

通过这些示例和实验，你将掌握BGE Rerank的使用方法，为构建高质量的检索系统打下基础。