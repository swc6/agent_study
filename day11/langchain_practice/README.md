# LangChain 基础链与 Prompt 模板练习项目

本目录包含LangChain基础链和Prompt模板的使用示例和实践代码，帮助你掌握LangChain的核心功能。

## 环境准备

### 安装依赖

```bash
# 进入langchain_practice目录
cd day11\langchain_practice

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

### 1. 基础链使用 (`basic_chains.py`)

演示LangChain基础链的使用：
- LLMChain：最基本的链，将提示词模板和LLM组合
- SimpleSequentialChain：按顺序执行多个链
- SequentialChain：支持多输入多输出的顺序链
- 批量处理：一次性处理多个输入

**运行方法**：
```bash
python basic_chains.py
```

### 2. Prompt模板编写 (`prompt_templates.py`)

展示如何编写和使用不同类型的Prompt模板：
- 基本Prompt模板：简单的变量替换
- 聊天Prompt模板：支持不同角色的消息
- 少样本Prompt模板：包含示例的模板
- 复杂Prompt模板：包含多行文本和结构
- 动态Prompt模板：根据参数动态生成

**运行方法**：
```bash
python prompt_templates.py
```

### 3. 链与模板集成 (`chain_template_integration.py`)

展示如何将链与模板集成，构建更复杂的应用：
- 问答链：回答用户问题
- 摘要链：生成文本摘要
- 翻译链：翻译文本
- 顺序链：组合多个链执行复杂任务
- 聊天链：支持多轮对话

**运行方法**：
```bash
python chain_template_integration.py
```

## 学习路径

1. **基础篇**：运行`basic_chains.py`，了解LangChain的基础链
2. **模板篇**：运行`prompt_templates.py`，学习Prompt模板的编写
3. **集成篇**：运行`chain_template_integration.py`，学习链与模板的集成
4. **实践篇**：构建自己的LangChain应用

## 最佳实践

### Prompt模板设计

1. **明确任务**：
   - 清楚地描述任务要求
   - 提供具体的输出格式
   - 设定合理的长度限制

2. **提供上下文**：
   - 包含必要的背景信息
   - 提供相关的示例
   - 说明输入数据的来源

3. **控制输出**：
   - 使用明确的指令
   - 设定输出格式
   - 限制输出范围

4. **优化提示词**：
   - 简洁明了，避免冗余
   - 层次分明，逻辑清晰
   - 适应模型的上下文窗口

### 链的设计

1. **模块化**：
   - 将复杂任务分解为简单步骤
   - 每个链只负责一个明确的功能
   - 便于测试和调试

2. **错误处理**：
   - 添加异常捕获
   - 实现重试机制
   - 设计降级策略

3. **性能优化**：
   - 合理设置批处理大小
   - 启用缓存减少重复调用
   - 优化模型参数

4. **可观测性**：
   - 添加日志记录
   - 实现监控指标
   - 设计调试工具

## 扩展实验

尝试以下实验来深入学习：

1. **自定义链**：创建自己的自定义链，实现特定功能
2. **复杂工作流程**：构建包含多个步骤的复杂工作流程
3. **多模型集成**：集成不同的LLM模型
4. **工具集成**：将外部工具集成到链中
5. **实时应用**：构建一个简单的Web服务

## 参考资源

- [LangChain官方文档](https://docs.langchain.com/)
- [LangChain GitHub仓库](https://github.com/langchain-ai/langchain)
- [LangChain Prompt模板文档](https://docs.langchain.com/guides/templates)
- [LangChain Chains文档](https://docs.langchain.com/guides/chains)

## 依赖说明

- `langchain`：LangChain核心框架
- `langchain-openai`：OpenAI集成
- `python-dotenv`：环境变量管理

通过这些示例和实验，你将掌握LangChain的基础链和Prompt模板的使用方法，为构建复杂的AI应用打下基础。