# LangChain 工具封装练习项目

本目录包含LangChain工具封装的使用示例和实践代码，帮助你掌握工具封装的核心技术。

## 环境准备

### 安装依赖

```bash
# 进入tool_practice目录
cd day12\tool_practice

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

### 1. 基础工具封装 (`basic_tools.py`)

演示基础工具的封装方法：
- WeatherTool：天气查询工具
- CalculatorTool：计算器工具
- SearchTool：搜索工具
- 使用Pydantic模型定义参数

**运行方法**：
```bash
python basic_tools.py
```

### 2. RAG工具封装 (`rag_tool.py`)

展示如何将RAG系统封装为工具：
- SimpleRAGTool：简单RAG工具，只返回检索到的文档
- RAGTool：高级RAG工具，返回检索结果和生成的答案
- 与FAISS向量存储集成

**运行方法**：
```bash
python rag_tool.py
```

### 3. 多工具集成 (`multi_tool_integration.py`)

展示如何集成多个工具并与智能体配合使用：
- MultiToolAgent：集成所有工具的智能体
- ToolSelector：根据任务类型选择工具的选择器
- 测试复杂任务的处理

**运行方法**：
```bash
python multi_tool_integration.py
```

## 学习路径

1. **基础篇**：运行`basic_tools.py`，了解基础工具的封装方法
2. **RAG篇**：运行`rag_tool.py`，学习如何封装RAG系统为工具
3. **集成篇**：运行`multi_tool_integration.py`，学习多工具的集成和智能体的使用
4. **实践篇**：构建自己的工具和智能体

## 工具封装最佳实践

### 1. 工具设计

- **明确的工具描述**：清晰、准确地描述工具的用途
- **合理的参数设计**：使用Pydantic模型定义参数
- **完善的错误处理**：捕获并处理异常
- **性能优化**：优化执行速度，使用缓存机制

### 2. 工具集成

- **模块化设计**：将工具功能模块化，便于维护和扩展
- **标准化接口**：遵循LangChain的工具接口规范
- **文档和测试**：提供详细的文档和测试

### 3. 安全考虑

- **输入验证**：验证工具的输入参数，防止注入攻击
- **权限控制**：为工具设置适当的权限
- **数据安全**：保护敏感数据，加密传输和存储

## 扩展实验

尝试以下实验来深入学习：

1. **自定义工具**：创建自己的工具，实现特定功能
2. **工具链**：构建工具链，实现复杂任务
3. **工具选择器**：根据任务类型智能选择工具
4. **工具监控**：添加工具使用的监控和日志
5. **工具优化**：优化工具的性能和可靠性

## 参考资源

- [LangChain官方文档](https://docs.langchain.com/)
- [LangChain Tools文档](https://docs.langchain.com/guides/tools)
- [Pydantic官方文档](https://pydantic-docs.helpmanual.io/)
- [DashScope官方文档](https://help.aliyun.com/product/10143404.html)

## 依赖说明

- `langchain`：LangChain核心框架
- `langchain-community`：LangChain社区工具
- `sentence-transformers`：提供BGE Embedding
- `faiss-cpu`：提供向量存储
- `python-dotenv`：环境变量管理
- `dashscope`：阿里云DashScope API

通过这些示例和实验，你将掌握LangChain工具封装的核心技术，为构建强大的智能体应用打下基础。