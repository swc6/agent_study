# LangChain 基础链与 Prompt 模板编写学习指南

## 1. LangChain 概述

LangChain 是一个用于构建基于大语言模型（LLM）应用的框架，它提供了一系列工具和组件，使开发者能够更轻松地创建复杂的AI应用。LangChain的核心概念包括链（Chains）、提示词模板（Prompt Templates）、工具（Tools）、记忆（Memory）和智能体（Agents）等。

### 1.1 LangChain 的价值

- **模块化**：提供了一系列可组合的组件
- **标准化**：统一了不同LLM的接口
- **扩展性**：易于集成外部工具和服务
- **可观测性**：提供了监控和调试工具
- **生态系统**：拥有丰富的集成和社区支持

### 1.2 核心概念

| 概念 | 描述 | 用途 |
|------|------|------|
| 链（Chains） | 将多个组件组合在一起的序列 | 构建复杂的工作流程 |
| 提示词模板（Prompt Templates） | 预定义的提示词结构 | 标准化提示词生成 |
| 工具（Tools） | 可供LLM使用的外部功能 | 扩展LLM的能力 |
| 记忆（Memory） | 存储对话历史的组件 | 支持多轮对话 |
| 智能体（Agents） | 基于LLM的决策系统 | 自主执行复杂任务 |

## 2. 基础链（Basic Chains）

基础链是LangChain中最基本的组件，它允许你将多个步骤组合在一起，形成一个完整的工作流程。

### 2.1 链的类型

**1. LLMChain**
- **用途**：最基本的链，将提示词模板和LLM组合在一起
- **适用场景**：简单的文本生成任务

**2. SequentialChain**
- **用途**：按顺序执行多个链
- **适用场景**：需要多步骤处理的任务

**3. SimpleSequentialChain**
- **用途**：SequentialChain的简化版，前一个链的输出作为后一个链的输入
- **适用场景**：线性工作流程

**4. RouterChain**
- **用途**：根据输入选择不同的链执行
- **适用场景**：需要分支逻辑的任务

### 2.2 基本链的使用

**LLMChain 示例**：

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 创建提示词模板
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请写一篇关于{topic}的短文，不少于200字。"
)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)

# 运行链
result = chain.run("人工智能的未来")
print(result)
```

**SequentialChain 示例**：

```python
from langchain.chains import SequentialChain

# 创建第一个链
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="请生成一个关于{topic}的标题。"
    ),
    output_key="title"
)

# 创建第二个链
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["title"],
        template="请根据标题 '{title}' 写一篇短文，不少于200字。"
    ),
    output_key="content"
)

# 创建顺序链
sequential_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["title", "content"]
)

# 运行链
result = sequential_chain.run("人工智能的未来")
print("标题:", result["title"])
print("内容:", result["content"])
```

### 2.3 链的配置与优化

**配置参数**：
- **temperature**：控制输出的随机性（0-1）
- **max_tokens**：控制输出的最大长度
- **top_p**：控制词汇的多样性
- **frequency_penalty**：控制重复词汇的惩罚
- **presence_penalty**：控制新主题的引入

**性能优化**：
- **批处理**：使用`batch()`方法处理多个输入
- **缓存**：启用LLM缓存减少重复调用
- **并行处理**：使用`async`方法并行执行多个链

## 3. Prompt 模板（Prompt Templates）

Prompt模板是LangChain中用于标准化提示词生成的组件，它允许你创建可重用的提示词结构，并通过变量动态填充内容。

### 3.1 模板类型

**1. PromptTemplate**
- **用途**：最基本的模板类型，支持简单的变量替换
- **适用场景**：简单的提示词生成

**2. ChatPromptTemplate**
- **用途**：专为聊天模型设计的模板，支持不同角色的消息
- **适用场景**：构建聊天应用

**3. FewShotPromptTemplate**
- **用途**：包含少量示例的模板，用于少样本学习
- **适用场景**：需要提供示例的任务

**4. SemanticSimilarityExampleSelector**
- **用途**：基于语义相似度选择示例的模板
- **适用场景**：需要动态选择示例的任务

### 3.2 模板语法

**基本语法**：
- 使用`{variable}`表示变量
- 支持条件表达式：`{% if condition %}...{% endif %}`
- 支持循环：`{% for item in items %}...{% endfor %}`
- 支持包含其他模板：`{% include "template_name" %}`

**示例**：

```python
from langchain.prompts import PromptTemplate

# 基本模板
prompt = PromptTemplate(
    input_variables=["topic", "length"],
    template="请写一篇关于{topic}的短文，不少于{length}字。"
)

# 使用模板
formatted_prompt = prompt.format(topic="人工智能", length="200")
print(formatted_prompt)
```

### 3.3 高级模板功能

**1. 消息模板**：

```python
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# 创建系统消息模板
system_template = SystemMessagePromptTemplate.from_template(
    "你是一个专业的{field}专家，回答问题要准确专业。"
)

# 创建人类消息模板
human_template = HumanMessagePromptTemplate.from_template(
    "请解释{topic}的基本原理。"
)

# 创建聊天提示词模板
chat_prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

# 格式化提示词
formatted_prompt = chat_prompt.format_prompt(
    field="人工智能",
    topic="机器学习"
).to_messages()

print(formatted_prompt)
```

**2. 少样本模板**：

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# 示例
examples = [
    {"input": "如何学习Python", "output": "学习Python的步骤：1. 学习基础语法 2. 实践小项目 3. 学习常用库 4. 参与开源项目"},
    {"input": "如何学习Java", "output": "学习Java的步骤：1. 学习基础语法 2. 理解面向对象 3. 学习Spring框架 4. 实践企业项目"}
]

# 示例模板
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="问题: {input}\n回答: {output}"
)

# 少样本提示词模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="请按照以下示例的格式回答问题：",
    suffix="问题: {input}\n回答:",
    input_variables=["input"]
)

# 格式化提示词
formatted_prompt = few_shot_prompt.format(input="如何学习人工智能")
print(formatted_prompt)
```

## 4. 链与模板的集成

将链与模板集成是LangChain的核心功能，它允许你构建复杂的工作流程。

### 4.1 自定义链

```python
from langchain.chains.base import Chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class CustomChain(Chain):
    """自定义链示例"""
    
    llm: ChatOpenAI
    prompt: PromptTemplate
    
    @property
    def input_keys(self):
        return self.prompt.input_variables
    
    @property
    def output_keys(self):
        return ["result"]
    
    def _call(self, inputs):
        # 格式化提示词
        formatted_prompt = self.prompt.format(**inputs)
        
        # 调用LLM
        response = self.llm.invoke(formatted_prompt)
        
        # 返回结果
        return {"result": response.content}

# 使用自定义链
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请写一篇关于{topic}的短文，不少于200字。"
)

chain = CustomChain(llm=llm, prompt=prompt)
result = chain.run("人工智能的未来")
print(result)
```

### 4.2 复杂工作流程

```python
from langchain.chains import SequentialChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 链1：生成标题
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="请为关于{topic}的文章生成5个吸引人的标题。"
    ),
    output_key="titles"
)

# 链2：选择最佳标题
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["titles"],
        template="请从以下标题中选择最佳的一个，并说明理由：\n{titles}"
    ),
    output_key="best_title"
)

# 链3：生成内容
chain3 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["best_title", "topic"],
        template="请根据标题 '{best_title}' 写一篇关于{topic}的文章，不少于500字。"
    ),
    output_key="content"
)

# 链4：生成摘要
chain4 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["content"],
        template="请为以下文章生成一个100字以内的摘要：\n{content}"
    ),
    output_key="summary"
)

# 组合成顺序链
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["topic"],
    output_variables=["titles", "best_title", "content", "summary"]
)

# 运行链
result = overall_chain.run("人工智能在医疗领域的应用")
print("标题列表:", result["titles"])
print("最佳标题:", result["best_title"])
print("文章内容:", result["content"])
print("摘要:", result["summary"])
```

## 5. 最佳实践

### 5.1 提示词设计

**1. 明确任务**：
- 清楚地描述任务要求
- 提供具体的输出格式
- 设定合理的长度限制

**2. 提供上下文**：
- 包含必要的背景信息
- 提供相关的示例
- 说明输入数据的来源

**3. 控制输出**：
- 使用明确的指令
- 设定输出格式
- 限制输出范围

**4. 优化提示词**：
- 简洁明了，避免冗余
- 层次分明，逻辑清晰
- 适应模型的上下文窗口

### 5.2 链的设计

**1. 模块化**：
- 将复杂任务分解为简单步骤
- 每个链只负责一个明确的功能
- 便于测试和调试

**2. 错误处理**：
- 添加异常捕获
- 实现重试机制
- 设计降级策略

**3. 性能优化**：
- 合理设置批处理大小
- 启用缓存减少重复调用
- 优化模型参数

**4. 可观测性**：
- 添加日志记录
- 实现监控指标
- 设计调试工具

### 5.3 实际应用建议

**1. 开发流程**：
- 从简单原型开始
- 逐步添加功能
- 持续测试和优化

**2. 部署策略**：
- 考虑使用容器化部署
- 实现负载均衡
- 设计容错机制

**3. 监控与维护**：
- 监控API调用频率和成本
- 跟踪模型性能
- 定期更新提示词和链

## 6. 常见问题与解决方案

### 6.1 提示词问题

**问题1：提示词过长**
- **原因**：超过模型的上下文窗口限制
- **解决方案**：
  - 缩短提示词
  - 分批次处理
  - 使用支持更长上下文的模型

**问题2：输出质量不稳定**
- **原因**：模型的随机性
- **解决方案**：
  - 降低temperature值
  - 提供更多示例
  - 增加输出约束

**问题3：模型不遵循指令**
- **原因**：提示词不够明确
- **解决方案**：
  - 使用更明确的指令
  - 提供示例
  - 增加指令的权重

### 6.2 链的问题

**问题1：链执行缓慢**
- **原因**：模型调用频繁或处理复杂
- **解决方案**：
  - 优化模型选择
  - 实现并行处理
  - 启用缓存

**问题2：链执行失败**
- **原因**：API错误或输入格式问题
- **解决方案**：
  - 添加错误处理
  - 实现重试机制
  - 验证输入格式

**问题3：内存使用过高**
- **原因**：处理大量数据或复杂链
- **解决方案**：
  - 优化批处理大小
  - 释放中间结果
  - 使用流式处理

### 6.3 集成问题

**问题1：与外部系统集成困难**
- **原因**：接口不兼容或数据格式问题
- **解决方案**：
  - 设计适配器
  - 标准化数据格式
  - 使用事件驱动架构

**问题2：部署环境差异**
- **原因**：开发环境与生产环境不同
- **解决方案**：
  - 使用容器化部署
  - 配置环境变量
  - 实现环境检测

**问题3：安全性问题**
- **原因**：API密钥管理或数据安全
- **解决方案**：
  - 使用环境变量存储密钥
  - 实现访问控制
  - 加密敏感数据

## 7. 案例分析

### 7.1 问答系统

**需求**：构建一个基于知识库的问答系统

**实现方案**：
1. **文档处理**：使用Docling解析文档
2. **文本切片**：使用RecursiveCharacterTextSplitter
3. **向量嵌入**：使用BGE Embedding
4. **向量存储**：使用FAISS或Milvus
5. **检索**：使用混合检索（BM25 + 向量）
6. **重排序**：使用BGE Rerank
7. **生成**：使用LLMChain生成答案

**关键代码**：

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 初始化组件
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
vectorstore = FAISS.load_local("faiss_index", embeddings)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# 运行链
result = qa_chain.run("什么是人工智能？")
print(result)
```

### 7.2 内容生成系统

**需求**：构建一个自动生成文章的系统

**实现方案**：
1. **主题分析**：使用LLMChain分析主题
2. **大纲生成**：使用LLMChain生成文章大纲
3. **内容生成**：使用LLMChain生成各部分内容
4. **内容整合**：使用LLMChain整合内容
5. **编辑优化**：使用LLMChain优化内容

**关键代码**：

```python
from langchain.chains import SequentialChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 链1：分析主题
chain1 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="请分析主题 '{topic}' 的核心内容和重要方面。"
    ),
    output_key="analysis"
)

# 链2：生成大纲
chain2 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic", "analysis"],
        template="基于主题 '{topic}' 和分析结果，生成一个详细的文章大纲。\n分析结果：{analysis}"
    ),
    output_key="outline"
)

# 链3：生成内容
chain3 = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic", "outline"],
        template="根据大纲生成关于 '{topic}' 的详细文章。\n大纲：{outline}"
    ),
    output_key="content"
)

# 组合成顺序链
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["analysis", "outline", "content"]
)

# 运行链
result = overall_chain.run("人工智能的未来发展")
print("分析:", result["analysis"])
print("大纲:", result["outline"])
print("内容:", result["content"])
```

## 8. 未来发展趋势

### 8.1 模型发展

**趋势1：更强大的模型**
- 更大的参数量
- 更长的上下文窗口
- 更好的多模态能力

**趋势2：更高效的模型**
- 模型压缩和蒸馏
- 量化技术
- 边缘设备部署

**趋势3：更专业的模型**
- 领域特定模型
- 任务特定模型
- 多语言模型

### 8.2 框架发展

**趋势1：更模块化的设计**
- 更细粒度的组件
- 更灵活的组合方式
- 更标准的接口

**趋势2：更智能的工具集成**
- 自动工具选择
- 工具使用优化
- 多工具协作

**趋势3：更强大的智能体**
- 更好的推理能力
- 更长的规划能力
- 更好的环境交互

### 8.3 应用发展

**趋势1：更个性化的应用**
- 基于用户历史的定制
- 实时适应用户需求
- 个性化的提示词和链

**趋势2：更复杂的任务**
- 多步骤任务处理
- 跨领域知识整合
- 实时决策和执行

**趋势3：更广泛的部署**
- 边缘设备部署
- 私有化部署
- 混合云部署

## 9. 总结

LangChain的基础链和Prompt模板是构建LLM应用的核心组件，它们提供了一种结构化、模块化的方式来组织和执行复杂的AI任务。通过本文的学习，你应该已经掌握了：

- LangChain的基本概念和核心组件
- 基础链的类型和使用方法
- Prompt模板的设计和应用
- 链与模板的集成方法
- 最佳实践和常见问题的解决方案
- 实际应用案例

在实际应用中，LangChain的灵活性和扩展性使其成为构建复杂AI应用的理想选择。随着模型技术和框架的不断发展，LangChain将在未来的AI应用开发中发挥越来越重要的作用。

通过不断学习和实践，你将能够使用LangChain构建更加智能、高效、可靠的AI应用，为各种领域的问题提供创新的解决方案。