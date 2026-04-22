# Docling文档解析练习

本目录包含使用Docling进行文档解析的练习代码。

## 环境准备

1. 安装Docling及依赖:
   ```bash
   pip install docling pypdf2 python-docx openpyxl
   ```

2. 准备测试文档:
   - 在当前目录添加一个 `example.pdf` 文件用于测试
   - 也可以使用其他格式的文档，如Word (.docx)或Excel (.xlsx)

## 使用方法

1. 运行练习脚本:
   ```bash
   python docling_demo.py
   ```

2. 脚本会执行以下操作:
   - 解析PDF文档
   - 提取基本信息（页数、文本长度）
   - 显示前2页的文本内容
   - 提取并显示表格（如果有）
   - 显示文档结构（标题、段落等）

## 练习内容

1. **基础练习**:
   - 尝试解析不同格式的文档（PDF、Word、Excel）
   - 观察解析结果，了解Docling的解析能力

2. **进阶练习**:
   - 修改脚本，尝试提取文档中的图片
   - 实现文档内容的结构化输出
   - 比较Docling与其他解析工具的性能

3. **挑战练习**:
   - 实现批量文档处理
   - 构建一个简单的文档信息提取系统
   - 与RAG系统集成，为后续步骤做准备

## 参考资源

- [Docling官方文档](https://docling.readthedocs.io/)
- [LangChain文档](https://docs.langchain.com/)
- [Python文档处理最佳实践](https://realpython.com/python-pdf/)

## 注意事项

- Docling在处理复杂文档时可能需要较长时间
- 对于扫描PDF，可能需要启用OCR功能
- 处理大型文档时，建议使用流式处理以节省内存