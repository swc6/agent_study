# 查询改写 (Query Rewrite) 练习项目

本目录包含查询改写的使用示例和实践代码，帮助你掌握查询改写的核心技术。

## 环境准备

### 安装依赖

```bash
# 进入query_rewrite_practice目录
cd day14\query_rewrite_practice

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

1. 创建`.env`文件，添加阿里云DashScope API密钥：
   ```
   DASHSCOPE_API_KEY=your_dashscope_api_key
   DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
   ```

2. 本项目使用Qwen模型，通过阿里云DashScope API调用。

## 示例代码

### 1. 基础查询改写 (`basic_query_rewrite.py`)

展示基本的查询改写功能：
- 将口语化查询转换为专业检索语句
- 批量处理多个查询
- 保持查询的核心意图

**运行方法**：
```bash
python basic_query_rewrite.py
```

### 2. 高级查询改写 (`advanced_query_rewrite.py`)

展示更高级的查询改写功能：
- 领域特定的查询改写
- 多轮查询改写（根据反馈优化）
- 更精确的查询转换

**运行方法**：
```bash
python advanced_query_rewrite.py
```

### 3. 查询改写与RAG集成 (`rag_integration.py`)

展示如何将查询改写与RAG系统集成：
- 带查询改写的RAG系统
- 不带查询改写的RAG系统（用于对比）
- 比较两种方法的检索效果

**运行方法**：
```bash
python rag_integration.py
```

## 学习路径

1. **基础篇**：运行`basic_query_rewrite.py`，了解查询改写的基本概念和实现
2. **进阶篇**：运行`advanced_query_rewrite.py`，学习领域特定和多轮查询改写
3. **实践篇**：运行`rag_integration.py`，学习如何将查询改写与RAG系统集成
4. **应用篇**：构建自己的查询改写系统，应用到实际项目中

## 查询改写的核心概念

### 1. 什么是查询改写？

查询改写（Query Rewrite）是指将用户的原始查询转换为更适合检索系统处理的形式，以提高检索的准确性和相关性。

### 2. 查询改写的价值

- **提高检索准确性**：将模糊的口语问题转换为精确的检索语句
- **增强用户体验**：允许用户使用自然语言提问
- **优化检索结果**：通过扩展或重构查询，提高相关文档的召回率
- **减少歧义**：消除查询中的歧义，提高检索的精度

### 3. 查询改写的方法

- **基于规则的方法**：使用预定义的规则进行改写
- **基于模板的方法**：使用预定义的模板进行改写
- **基于模型的方法**：使用大语言模型进行改写
- **混合方法**：结合多种方法进行改写

## 最佳实践

### 1. 提示词设计

- **明确任务**：清晰地描述查询改写的目标
- **提供示例**：包含成功的改写示例
- **领域知识**：融入领域专业知识
- **多轮改写**：进行多轮改写，逐步优化

### 2. 模型选择

- **小模型**：速度快，适合简单改写
- **大模型**：能力强，适合复杂改写
- **领域模型**：专业度高，适合特定领域

### 3. 评估与反馈

- **评估指标**：改写准确性、改写质量、检索效果、用户满意度
- **反馈机制**：收集用户反馈，分析改写效果，持续优化
- **A/B测试**：对比不同改写方法的效果

## 扩展实验

尝试以下实验来深入学习：

1. **多语言查询改写**：支持中文、英文等多语言查询
2. **跨领域查询改写**：将通用查询转换为特定领域的专业查询
3. **复杂查询改写**：处理包含多个意图的复杂查询
4. **实时查询改写**：根据对话上下文实时调整改写策略
5. **个性化查询改写**：根据用户的历史行为进行个性化改写

## 参考资源

- [LangChain官方文档](https://docs.langchain.com/)
- [DashScope官方文档](https://help.aliyun.com/product/10143404.html)
- [查询改写技术综述](https://arxiv.org/abs/2104.08773)
- [RAG系统中的查询改写](https://arxiv.org/abs/2305.00978)

## 依赖说明

- `langchain`：LangChain核心框架
- `langchain-community`：LangChain社区工具
- `python-dotenv`：环境变量管理
- `dashscope`：阿里云DashScope API
- `faiss-cpu`：向量存储
- `sentence-transformers`：文本嵌入

通过这些示例和实验，你将掌握查询改写的核心技术，为构建更智能、更高效的RAG系统打下基础。