# ReAct Agent 练习项目

本目录包含ReAct Agent的使用示例和实践代码，帮助你掌握ReAct Agent的核心技术。

## 环境准备

### 安装依赖

```bash
# 进入react_practice目录
cd day13\react_practice

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

### 1. 最简ReAct Agent (`simple_react_agent.py`)

展示ReAct Agent的基本实现：
- 实现思考-行动-观察循环
- 手动解析LLM响应
- 执行工具调用
- 管理对话历史

**运行方法**：
```bash
python simple_react_agent.py
```

### 2. LangChain ReAct Agent (`langchain_react_agent.py`)

展示使用LangChain实现ReAct Agent：
- 使用LangChain的Agent框架
- 集成多个工具
- 结构化的工具调用
- 详细的执行过程

**运行方法**：
```bash
python langchain_react_agent.py
```

### 3. 带记忆的ReAct Agent (`memory_react_agent.py`)

展示如何实现带记忆的ReAct Agent：
- 使用ConversationBufferMemory
- 支持多轮对话
- 保持上下文信息
- 连续的任务处理

**运行方法**：
```bash
python memory_react_agent.py
```

## 学习路径

1. **基础篇**：运行`simple_react_agent.py`，了解ReAct的基本原理
2. **进阶篇**：运行`langchain_react_agent.py`，学习使用LangChain实现ReAct Agent
3. **高级篇**：运行`memory_react_agent.py`，学习如何实现多轮对话
4. **实践篇**：构建自己的ReAct Agent系统

## ReAct Agent 核心概念

### 1. 思考-行动-观察循环

- **思考（Think）**：分析当前状态，规划下一步行动
- **行动（Act）**：执行计划的行动，与环境交互
- **观察（Observe）**：接收环境的反馈，了解行动的结果
- **循环**：基于新的观察结果，重复上述过程

### 2. 关键组件

- **大语言模型（LLM）**：作为智能体的大脑，负责推理和决策
- **工具（Tools）**：智能体与外部环境交互的接口
- **环境（Environment）**：智能体操作的外部世界
- **记忆（Memory）**：存储历史交互信息

### 3. 提示词设计

ReAct的提示词设计非常关键，它需要引导LLM生成特定格式的输出，包括思考、行动和观察。

## 最佳实践

### 1. 提示词设计

- **明确的格式**：指定清晰的输出格式
- **详细的工具描述**：清晰地描述工具的用途和参数
- **示例引导**：提供成功的思考-行动-观察示例
- **合理的约束**：设定适当的时间和迭代限制

### 2. 工具管理

- **工具选择**：根据任务选择合适的工具
- **工具描述**：提供详细的工具描述和参数说明
- **错误处理**：处理工具执行失败的情况
- **工具返回**：确保工具返回有用的信息

### 3. 状态管理

- **记忆管理**：合理设置记忆的大小和内容
- **迭代控制**：设置合理的最大迭代次数
- **状态更新**：及时更新状态信息
- **循环检测**：检测并避免循环行为

## 扩展实验

尝试以下实验来深入学习：

1. **自定义工具**：创建自己的工具，实现特定功能
2. **多工具协作**：集成多个工具，处理复杂任务
3. **记忆优化**：改进记忆管理，提高多轮对话质量
4. **提示词优化**：优化提示词，提高推理质量
5. **性能评估**：评估ReAct Agent的性能和准确性

## 参考资源

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [LangChain官方文档](https://docs.langchain.com/)
- [LangChain Agents文档](https://docs.langchain.com/guides/agents)
- [DashScope官方文档](https://help.aliyun.com/product/10143404.html)

## 依赖说明

- `langchain`：LangChain核心框架
- `langchain-community`：LangChain社区工具
- `python-dotenv`：环境变量管理
- `dashscope`：阿里云DashScope API

通过这些示例和实验，你将掌握ReAct Agent的核心技术，为构建强大的智能体系统打下基础。