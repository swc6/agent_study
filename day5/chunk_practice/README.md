# 文本切片（Chunk）案例学习

本目录包含多种文本切片策略的实现和示例，帮助你理解不同切片方法的原理和应用。

## 环境准备

```bash
# 安装必要的依赖
pip install langchain
```

## 1. 固定长度切片（Fixed-size Chunking）

### 实现代码

```python
# fixed_size_chunking.py
def fixed_size_chunking(text, chunk_size=1000, overlap=200):
    """
    固定长度切片实现
    
    参数:
        text: 要切片的文本
        chunk_size: 切片大小（字符数）
        overlap: 重叠部分大小（字符数）
    
    返回:
        切片后的文本列表
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 测试示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    
    人工智能的发展历史可以分为几个重要阶段：
    1. 孕育期（1950-1956）：图灵提出了著名的图灵测试，为人工智能的发展奠定了理论基础。
    2. 诞生期（1956-1970）：达特茅斯会议标志着人工智能的正式诞生。
    3. 发展期（1970-1990）：专家系统的出现推动了人工智能的发展。
    4. 繁荣期（1990至今）：机器学习、深度学习等技术的发展使得人工智能取得了重大突破。
    
    人工智能的应用领域非常广泛，包括：
    - 自然语言处理：机器翻译、语音识别、文本分析等
    - 计算机视觉：图像识别、目标检测、人脸识别等
    - 机器人技术：工业机器人、服务机器人、教育机器人等
    - 推荐系统：个性化推荐、内容推荐等
    - 自动驾驶：自动车辆、无人机等
    
    未来，人工智能将继续发展，在更多领域发挥重要作用，同时也需要关注伦理、安全等问题。
    """
    
    # 测试固定长度切片
    chunks = fixed_size_chunking(sample_text, chunk_size=300, overlap=50)
    
    print(f"原始文本长度: {len(sample_text)} 字符")
    print(f"切片数量: {len(chunks)}")
    print("\n切片结果:")
    for i, chunk in enumerate(chunks):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk)
```

### 运行结果

```bash
python fixed_size_chunking.py
```

## 2. 基于段落的切片

### 实现代码

```python
# paragraph_chunking.py
def paragraph_chunking(text):
    """
    基于段落的切片实现
    
    参数:
        text: 要切片的文本
    
    返回:
        切片后的文本列表
    """
    # 按换行符分割段落
    paragraphs = text.split('\n\n')
    # 过滤空段落
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

# 测试示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    
    人工智能的发展历史可以分为几个重要阶段：
    1. 孕育期（1950-1956）：图灵提出了著名的图灵测试，为人工智能的发展奠定了理论基础。
    2. 诞生期（1956-1970）：达特茅斯会议标志着人工智能的正式诞生。
    3. 发展期（1970-1990）：专家系统的出现推动了人工智能的发展。
    4. 繁荣期（1990至今）：机器学习、深度学习等技术的发展使得人工智能取得了重大突破。
    
    人工智能的应用领域非常广泛，包括：
    - 自然语言处理：机器翻译、语音识别、文本分析等
    - 计算机视觉：图像识别、目标检测、人脸识别等
    - 机器人技术：工业机器人、服务机器人、教育机器人等
    - 推荐系统：个性化推荐、内容推荐等
    - 自动驾驶：自动车辆、无人机等
    
    未来，人工智能将继续发展，在更多领域发挥重要作用，同时也需要关注伦理、安全等问题。
    """
    
    # 测试段落切片
    chunks = paragraph_chunking(sample_text)
    
    print(f"段落数量: {len(chunks)}")
    print("\n切片结果:")
    for i, chunk in enumerate(chunks):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk)
```

## 3. LangChain RecursiveCharacterTextSplitter

### 实现代码

```python
# langchain_chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 测试示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    
    人工智能的发展历史可以分为几个重要阶段：
    1. 孕育期（1950-1956）：图灵提出了著名的图灵测试，为人工智能的发展奠定了理论基础。
    2. 诞生期（1956-1970）：达特茅斯会议标志着人工智能的正式诞生。
    3. 发展期（1970-1990）：专家系统的出现推动了人工智能的发展。
    4. 繁荣期（1990至今）：机器学习、深度学习等技术的发展使得人工智能取得了重大突破。
    
    人工智能的应用领域非常广泛，包括：
    - 自然语言处理：机器翻译、语音识别、文本分析等
    - 计算机视觉：图像识别、目标检测、人脸识别等
    - 机器人技术：工业机器人、服务机器人、教育机器人等
    - 推荐系统：个性化推荐、内容推荐等
    - 自动驾驶：自动车辆、无人机等
    
    未来，人工智能将继续发展，在更多领域发挥重要作用，同时也需要关注伦理、安全等问题。
    """
    
    # 初始化RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,          # 切片大小
        chunk_overlap=50,        # 重叠部分大小
        length_function=len,      # 长度计算函数
        separators=["\n\n", "\n", " ", ""]  # 分割符优先级
    )
    
    # 执行切片
    chunks = text_splitter.split_text(sample_text)
    
    print(f"切片数量: {len(chunks)}")
    print("\n切片结果:")
    for i, chunk in enumerate(chunks):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk)
```

## 4. 语义切片（基于相似度）

### 实现代码

```python
# semantic_chunking.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 模拟嵌入函数（实际应用中应使用真实的嵌入模型）
def mock_embedding(text):
    """模拟文本嵌入"""
    # 简单模拟，实际应用中应使用BGE等嵌入模型
    return np.random.rand(768)

def semantic_chunking(text, max_chunk_size=300, similarity_threshold=0.7):
    """
    基于语义相似度的切片实现
    
    参数:
        text: 要切片的文本
        max_chunk_size: 最大切片大小
        similarity_threshold: 相似度阈值
    
    返回:
        切片后的文本列表
    """
    # 按句子分割
    sentences = text.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # 如果当前块为空，直接添加
        if not current_chunk:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # 检查相似度（这里使用模拟值，实际应计算真实相似度）
            similarity = np.random.uniform(0.5, 1.0)  # 模拟相似度
            
            # 如果相似度高且不超过最大长度，添加到当前块
            if similarity > similarity_threshold and current_length + sentence_length < max_chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 否则，创建新块
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 测试示例
if __name__ == "__main__":
    # 示例文本
    sample_text = """
    人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    人工智能的发展历史可以分为几个重要阶段。
    孕育期（1950-1956）：图灵提出了著名的图灵测试，为人工智能的发展奠定了理论基础。
    诞生期（1956-1970）：达特茅斯会议标志着人工智能的正式诞生。
    发展期（1970-1990）：专家系统的出现推动了人工智能的发展。
    繁荣期（1990至今）：机器学习、深度学习等技术的发展使得人工智能取得了重大突破。
    人工智能的应用领域非常广泛，包括自然语言处理、计算机视觉、机器人技术等。
    自然语言处理包括机器翻译、语音识别、文本分析等。
    计算机视觉包括图像识别、目标检测、人脸识别等。
    机器人技术包括工业机器人、服务机器人、教育机器人等。
    """
    
    # 测试语义切片
    chunks = semantic_chunking(sample_text, max_chunk_size=300, similarity_threshold=0.7)
    
    print(f"切片数量: {len(chunks)}")
    print("\n切片结果:")
    for i, chunk in enumerate(chunks):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk)
```

## 5. 综合示例：不同切片策略对比

### 实现代码

```python
# chunking_comparison.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 固定长度切片
def fixed_size_chunking(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 基于段落的切片
def paragraph_chunking(text):
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

# 测试不同切片策略
if __name__ == "__main__":
    # 加载示例文本
    with open('sample_document.txt', 'r', encoding='utf-8') as f:
        sample_text = f.read()
    
    print(f"原始文本长度: {len(sample_text)} 字符")
    print("\n" + "="*60)
    
    # 1. 固定长度切片
    print("\n1. 固定长度切片:")
    fixed_chunks = fixed_size_chunking(sample_text, chunk_size=500, overlap=100)
    print(f"切片数量: {len(fixed_chunks)}")
    for i, chunk in enumerate(fixed_chunks[:2]):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk[:100] + "...")
    
    # 2. 基于段落的切片
    print("\n" + "="*60)
    print("\n2. 基于段落的切片:")
    para_chunks = paragraph_chunking(sample_text)
    print(f"切片数量: {len(para_chunks)}")
    for i, chunk in enumerate(para_chunks[:2]):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk[:100] + "...")
    
    # 3. LangChain RecursiveCharacterTextSplitter
    print("\n" + "="*60)
    print("\n3. LangChain RecursiveCharacterTextSplitter:")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          
        chunk_overlap=100,        
        separators=["\n\n", "\n", " ", ""]
    )
    langchain_chunks = text_splitter.split_text(sample_text)
    print(f"切片数量: {len(langchain_chunks)}")
    for i, chunk in enumerate(langchain_chunks[:2]):
        print(f"\n切片 {i+1} (长度: {len(chunk)}):")
        print(chunk[:100] + "...")
```

### 示例文档

创建一个 `sample_document.txt` 文件，包含较长的文本内容：

```
# 人工智能发展概述

人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。

## 发展历史

人工智能的发展历史可以分为几个重要阶段：

### 1. 孕育期（1950-1956）
图灵提出了著名的图灵测试，为人工智能的发展奠定了理论基础。1950年，阿兰·图灵发表了题为《计算机器与智能》的论文，提出了判断机器是否具有智能的标准，即图灵测试。

### 2. 诞生期（1956-1970）
1956年，达特茅斯会议标志着人工智能的正式诞生。在这次会议上，约翰·麦卡锡首次提出了"人工智能"这一术语，并组织了第一次人工智能研讨会。

### 3. 发展期（1970-1990）
专家系统的出现推动了人工智能的发展。这一时期，人工智能研究主要集中在专家系统、自然语言处理和计算机视觉等领域。

### 4. 繁荣期（1990至今）
机器学习、深度学习等技术的发展使得人工智能取得了重大突破。特别是2012年以来，深度学习在图像识别、语音识别等领域取得了革命性的成果。

## 应用领域

人工智能的应用领域非常广泛，包括：

### 1. 自然语言处理
- 机器翻译：如Google翻译、DeepL等
- 语音识别：如Siri、Alexa等智能助手
- 文本分析：情感分析、文本分类、信息提取等

### 2. 计算机视觉
- 图像识别：物体识别、场景理解等
- 目标检测：行人检测、车辆检测等
- 人脸识别：身份验证、表情分析等

### 3. 机器人技术
- 工业机器人：自动化生产线、装配机器人等
- 服务机器人：酒店服务、医疗护理等
- 教育机器人：智能辅导、编程教育等

### 4. 推荐系统
- 个性化推荐：电商推荐、内容推荐等
- 搜索排序：搜索引擎结果排序
- 广告投放：精准广告推荐

### 5. 自动驾驶
- 自动车辆：特斯拉Autopilot、Waymo等
- 无人机：物流配送、农业监测等
- 智能交通：交通管理、拥堵预测等

## 挑战与未来

人工智能的发展面临着诸多挑战，包括：

1. 数据隐私：如何保护用户数据的安全和隐私
2. 算法偏见：如何避免算法中的偏见和歧视
3. 伦理问题：如何确保人工智能的使用符合伦理道德
4. 就业影响：人工智能对就业市场的影响
5. 安全风险：人工智能系统的安全性和可靠性

未来，人工智能将继续发展，在更多领域发挥重要作用。随着技术的不断进步，人工智能有望在医疗、教育、环境等领域带来更多创新和突破。

同时，我们也需要关注人工智能的伦理、法律和社会影响，确保人工智能的发展符合人类的利益和价值观。
```

## 6. 运行方法

1. 进入chunk_practice目录
2. 安装依赖：`pip install langchain`
3. 运行各个示例：
   ```bash
   python fixed_size_chunking.py
   python paragraph_chunking.py
   python langchain_chunking.py
   python semantic_chunking.py
   python chunking_comparison.py
   ```

## 7. 实验建议

1. **调整参数**：尝试不同的chunk_size和overlap值，观察切片结果的变化
2. **比较策略**：对比不同切片策略的效果，分析各自的优缺点
3. **实际应用**：使用真实文档进行测试，评估切片效果
4. **性能测试**：测试不同切片策略的处理速度

通过这些实验，你将更好地理解文本切片的原理和应用，为构建高质量的RAG系统打下基础。