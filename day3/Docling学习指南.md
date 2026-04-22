# Docling学习指南

## 1. Docling简介

Docling是一个强大的文档解析工具，专为高精度文档处理而设计。它能够处理多种格式的文档，包括PDF、Word、Excel等，并且能够保留文档的结构和格式信息。

### 主要特点

- **高精度解析**：能够准确提取文本、表格、图像等内容
- **多格式支持**：支持PDF、Word、Excel等多种文档格式
- **结构保留**：保留文档的层次结构和排版信息
- **处理复杂文档**：能够处理包含复杂格式的文档
- **易于集成**：提供简洁的API接口

## 2. 安装与配置

### 基本安装

```bash
# 安装Docling
pip install docling

# 安装依赖
pip install pypdf2 python-docx openpyxl
```

### 验证安装

```python
import docling
print("Docling版本:", docling.__version__)
```

## 3. 基本使用

### 解析PDF文档

```python
from docling import Document

# 解析PDF文档
doc = Document("example.pdf")

# 提取文本
text = doc.text
print("提取的文本:", text[:500])  # 打印前500个字符

# 提取页面
for i, page in enumerate(doc.pages):
    print(f"页面 {i+1}:", page.text[:200])

# 提取表格
tables = doc.tables
for i, table in enumerate(tables):
    print(f"表格 {i+1}:")
    print(table.to_markdown())
```

### 解析Word文档

```python
from docling import Document

# 解析Word文档
doc = Document("example.docx")

# 提取文本
text = doc.text
print("提取的文本:", text[:500])

# 提取段落
for i, paragraph in enumerate(doc.paragraphs):
    print(f"段落 {i+1}:", paragraph.text)

# 提取表格
tables = doc.tables
for i, table in enumerate(tables):
    print(f"表格 {i+1}:")
    print(table.to_markdown())
```

### 解析Excel文档

```python
from docling import Document

# 解析Excel文档
doc = Document("example.xlsx")

# 提取工作表
for sheet_name in doc.sheet_names:
    print(f"工作表: {sheet_name}")
    sheet = doc[sheet_name]
    print(sheet.to_markdown())
```

## 4. 高级功能

### 提取文档结构

```python
from docling import Document

# 解析文档
doc = Document("example.pdf")

# 提取文档结构
print("文档结构:")
for element in doc.structure:
    if element.type == "heading":
        print(f"标题 {element.level}: {element.text}")
    elif element.type == "paragraph":
        print(f"段落: {element.text[:100]}...")
    elif element.type == "table":
        print(f"表格: {element.caption if element.caption else '无标题'}")
```

### 处理扫描PDF

```python
from docling import Document

# 解析扫描PDF（需要OCR支持）
doc = Document("scanned.pdf", ocr=True)

# 提取文本
text = doc.text
print("提取的文本:", text[:500])
```

### 批量处理

```python
import os
from docling import Document

# 批量处理目录中的文档
directory = "documents"
for filename in os.listdir(directory):
    if filename.endswith((".pdf", ".docx", ".xlsx")):
        filepath = os.path.join(directory, filename)
        print(f"处理文件: {filename}")
        try:
            doc = Document(filepath)
            print(f"  页数: {len(doc.pages)}")
            print(f"  表格数: {len(doc.tables)}")
            print(f"  文本长度: {len(doc.text)}")
        except Exception as e:
            print(f"  错误: {str(e)}")
```

## 5. 与其他工具对比

| 工具 | 优势 | 劣势 |
|------|------|------|
| Docling | 高精度解析、多格式支持、结构保留 | 安装依赖较多 |
| PyPDF2 | 轻量级、易于安装 | 解析精度较低、不支持Word/Excel |
| pdfplumber | 表格提取能力强 | 只支持PDF |
| python-docx | 专门针对Word文档 | 只支持Word |
| openpyxl | 专门针对Excel文档 | 只支持Excel |

## 6. 最佳实践

### 1. 处理大文档

对于大型文档，建议使用流式处理：

```python
from docling import Document

# 流式处理大文档
doc = Document("large_document.pdf", stream=True)

# 逐页处理
for page in doc.pages:
    # 处理当前页面
    text = page.text
    # 进行后续处理...
```

### 2. 处理复杂表格

对于复杂表格，建议使用表格的结构化表示：

```python
from docling import Document

# 解析文档
doc = Document("document_with_tables.pdf")

# 处理表格
for table in doc.tables:
    # 获取表格的行列数
    rows, cols = table.shape
    print(f"表格大小: {rows}行 x {cols}列")
    
    # 访问表格单元格
    for i in range(rows):
        for j in range(cols):
            cell_value = table[i, j]
            print(f"单元格 ({i+1},{j+1}): {cell_value}")
    
    # 转换为DataFrame
    df = table.to_dataframe()
    print(df)
```

### 3. 处理多栏文档

对于多栏布局的文档，Docling能够正确处理：

```python
from docling import Document

# 解析多栏文档
doc = Document("two_column_document.pdf")

# 提取文本（会按照阅读顺序）
text = doc.text
print(text)
```

## 7. 常见问题与解决方案

### 1. 解析速度慢

**原因**：文档较大或包含复杂内容

**解决方案**：
- 使用流式处理
- 只提取需要的部分
- 考虑使用多线程处理

### 2. 解析结果乱码

**原因**：文档编码问题或字体缺失

**解决方案**：
- 确保文档编码正确
- 安装缺失的字体
- 尝试使用OCR模式

### 3. 表格提取不准确

**原因**：表格结构复杂或扫描质量差

**解决方案**：
- 调整解析参数
- 使用OCR模式（对于扫描文档）
- 手动后处理表格数据

## 8. 实践项目

### 项目：文档信息提取系统

**目标**：从多种格式的文档中提取关键信息

**步骤**：

1. **安装依赖**：
   ```bash
   pip install docling pandas
   ```

2. **创建提取脚本**：
   ```python
   import os
   import pandas as pd
   from docling import Document

   def extract_information(filepath):
       """从文档中提取关键信息"""
       doc = Document(filepath)
       
       # 提取基本信息
       info = {
           "文件名": os.path.basename(filepath),
           "文件类型": os.path.splitext(filepath)[1],
           "文本长度": len(doc.text),
           "页数": len(doc.pages) if hasattr(doc, 'pages') else 1,
           "表格数": len(doc.tables) if hasattr(doc, 'tables') else 0,
       }
       
       # 提取标题
       if hasattr(doc, 'structure'):
           headings = []
           for element in doc.structure:
               if element.type == "heading":
                   headings.append(f"{'#' * element.level} {element.text}")
           info["标题"] = "\n".join(headings)
       
       return info

   # 处理目录中的所有文档
   directory = "documents"
   results = []
   
   for filename in os.listdir(directory):
       if filename.endswith((".pdf", ".docx", ".xlsx")):
           filepath = os.path.join(directory, filename)
           try:
               info = extract_information(filepath)
               results.append(info)
           except Exception as e:
               print(f"处理{filename}时出错: {str(e)}")

   # 保存结果到Excel
   df = pd.DataFrame(results)
   df.to_excel("document_info.xlsx", index=False)
   print("信息提取完成，结果保存到 document_info.xlsx")
   ```

3. **运行脚本**：
   ```bash
   python extract_info.py
   ```

4. **查看结果**：
   打开生成的 `document_info.xlsx` 文件查看提取的信息

## 9. 总结

Docling是一个功能强大的文档解析工具，能够处理多种格式的文档并保留其结构信息。通过本指南的学习，你应该能够：

- 安装和配置Docling
- 解析不同格式的文档
- 提取文本、表格等内容
- 处理复杂的文档结构
- 应用Docling到实际项目中

Docling将成为你构建RAG系统的重要工具，为后续的文本切片和向量嵌入提供高质量的输入数据。