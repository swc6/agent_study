# 简单Agent实现（使用Qwen 3.5）

本项目实现了一个基于ReAct模式的简单Agent，使用阿里云的Qwen 3.5大语言模型。

## 功能特性

- 使用阿里云Qwen 3.5大语言模型
- 实现ReAct思考-行动-观察循环
- 支持搜索和计算两种工具
- 支持多轮对话

## 环境要求

- Python 3.9+
- 阿里云DashScope API账号

## 安装步骤

1. 克隆项目到本地

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置API密钥：
   - 编辑`.env`文件，填写你的阿里云DashScope API密钥
   ```
   DASHSCOPE_API_KEY=your_dashscope_api_key
   ```

## 使用方法

运行Agent测试：

```bash
python agent.py
```

## 代码结构

- `agent.py`：主要实现文件，包含Agent的核心逻辑
- `.env`：环境变量配置文件，存储API密钥
- `requirements.txt`：依赖包列表

## 工具说明

1. **search**：搜索工具，用于获取信息
   - 参数：查询语句
   - 返回：搜索结果

2. **calculate**：计算工具，用于计算数学表达式
   - 参数：数学表达式
   - 返回：计算结果

## 测试示例

运行后会执行三个测试：
1. 测试搜索ReAct模式的信息
2. 测试计算数学表达式
3. 测试多轮对话功能

## 扩展建议

- 添加更多工具，如文件操作、网络请求等
- 优化提示词，提高Agent的决策质量
- 添加记忆系统，增强多轮对话能力
- 集成RAG系统，提高知识准确性

## 注意事项

- 确保你的阿里云DashScope API账号有足够的调用额度
- 对于生产环境，建议添加错误处理和异常捕获
- 可以根据具体场景调整模型参数和工具实现