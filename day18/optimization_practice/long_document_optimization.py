# 长文档优化示例

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os

# 加载长文档
def load_long_document(file_path):
    """加载长文档"""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        return documents[0]
    except Exception as e:
        print(f"加载文档失败: {str(e)}")
        return None

# 分块长文档
def chunk_long_document(document, chunk_size=1000, chunk_overlap=200):
    """分块长文档"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents([document])
    return chunks

# 提取关键信息
def extract_key_information(document, max_length=2000):
    """提取文档中的关键信息"""
    content = document.page_content
    
    # 提取标题和小标题
    lines = content.split('\n')
    key_info = []
    
    for line in lines:
        line = line.strip()
        if line:
            # 假设标题以#开头
            if line.startswith('#'):
                key_info.append(line)
            # 假设重要信息包含关键词
            elif any(keyword in line for keyword in ['重要', '关键', '核心', '结论', '摘要']):
                key_info.append(line)
    
    # 合并关键信息
    key_info_text = '\n'.join(key_info)
    
    # 如果关键信息不足，添加文档开头部分
    if len(key_info_text) < max_length:
        intro = content[:max_length - len(key_info_text)]
        key_info_text = intro + '\n' + key_info_text
    
    return key_info_text[:max_length]

# 层次化处理长文档
def process_document_hierarchically(document):
    """层次化处理长文档"""
    # 提取文档级信息
    doc_level_info = extract_key_information(document)
    
    # 分块处理
    chunks = chunk_long_document(document)
    
    # 为每个块提取关键信息
    chunk_level_info = []
    for i, chunk in enumerate(chunks):
        chunk_info = extract_key_information(chunk, max_length=500)
        chunk_level_info.append((i, chunk_info))
    
    return doc_level_info, chunk_level_info

# 测试长文档优化
def test_long_document_optimization():
    """测试长文档优化"""
    print("=== 测试长文档优化 ===")
    
    # 创建一个示例长文档
    sample_content = """
# 人工智能概述

## 1. 什么是人工智能

人工智能（Artificial Intelligence，AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能的发展可以追溯到20世纪50年代，当时计算机科学家开始探索如何让机器模拟人类的智能行为。

### 1.1 人工智能的定义

人工智能是指由人工制造出来的系统所表现出来的智能。通常，人工智能是指通过普通计算机程序来呈现人类智能的技术。

### 1.2 人工智能的历史

人工智能的发展经历了几个重要阶段：

- **早期阶段（1950s-1970s）**：人工智能概念的提出和早期研究，包括图灵测试的提出。
- **瓶颈阶段（1970s-1980s）**：由于计算能力和数据的限制，人工智能研究进入瓶颈期。
- **专家系统时代（1980s-1990s）**：专家系统的出现和应用，人工智能开始在特定领域发挥作用。
- **机器学习时代（1990s-2010s）**：机器学习算法的发展，特别是支持向量机和决策树等算法的应用。
- **深度学习时代（2010s至今）**：深度学习的突破，特别是卷积神经网络和循环神经网络的应用，使得人工智能在图像识别、自然语言处理等领域取得重大进展。

## 2. 人工智能的技术分支

### 2.1 机器学习

机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。机器学习算法可以分为监督学习、无监督学习和强化学习等类型。

### 2.2 深度学习

深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、语音识别、自然语言处理等领域取得了显著的成果。

### 2.3 自然语言处理

自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。自然语言处理的应用包括机器翻译、情感分析、文本摘要等。

### 2.4 计算机视觉

计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。计算机视觉的应用包括图像识别、目标检测、人脸识别等。

## 3. 人工智能的应用

### 3.1 医疗领域

人工智能在医疗领域的应用包括疾病诊断、药物发现、医疗影像分析等。例如，人工智能可以帮助医生更准确地诊断疾病，预测疾病的发展趋势，以及制定个性化的治疗方案。

### 3.2 金融领域

人工智能在金融领域的应用包括风险评估、 fraud detection、算法交易等。例如，人工智能可以帮助金融机构评估贷款风险，检测欺诈行为，以及优化投资策略。

### 3.3 交通领域

人工智能在交通领域的应用包括自动驾驶、交通流量管理、智能导航等。例如，人工智能可以帮助车辆实现自动驾驶，优化交通流量，以及提供实时的导航建议。

### 3.4 教育领域

人工智能在教育领域的应用包括个性化学习、智能辅导、自动评估等。例如，人工智能可以根据学生的学习情况提供个性化的学习内容，提供智能辅导，以及自动评估学生的作业和考试。

## 4. 人工智能的挑战和未来

### 4.1 挑战

人工智能面临的挑战包括：

- **数据质量和数量**：人工智能需要大量高质量的数据来训练模型。
- **计算资源**：深度学习模型需要大量的计算资源。
- **可解释性**：人工智能模型的决策过程往往难以解释。
- **伦理和隐私**：人工智能的应用可能涉及伦理和隐私问题。
- **安全性**：人工智能系统可能被攻击或滥用。

### 4.2 未来发展趋势

人工智能的未来发展趋势包括：

- **多模态融合**：结合文本、图像、音频等多种模态的信息。
- **自主学习**：使人工智能系统能够自主学习和适应新的环境。
- **人机协作**：人工智能与人类的协作将变得更加紧密。
- **边缘计算**：人工智能将更多地部署在边缘设备上。
- **量子计算**：量子计算可能为人工智能带来新的突破。

## 5. 结论

人工智能是一项快速发展的技术，它已经在各个领域产生了深远的影响。随着技术的不断进步，人工智能将在更多领域发挥重要作用，为人类社会带来更多的便利和福祉。然而，我们也需要关注人工智能带来的挑战，确保其发展符合人类的利益和价值观。
    """
    
    # 保存示例文档
    with open("sample_long_document.txt", 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    # 加载文档
    document = load_long_document("sample_long_document.txt")
    
    if document:
        print(f"文档长度: {len(document.page_content)} 字符")
        
        # 测试分块
        print("\n=== 测试文档分块 ===")
        chunks = chunk_long_document(document, chunk_size=500, chunk_overlap=100)
        print(f"分块数量: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n块 {i+1} (长度: {len(chunk.page_content)}):")
            print(chunk.page_content[:100] + "...")
        
        # 测试关键信息提取
        print("\n=== 测试关键信息提取 ===")
        key_info = extract_key_information(document)
        print(f"提取的关键信息 (长度: {len(key_info)}):")
        print(key_info)
        
        # 测试层次化处理
        print("\n=== 测试层次化处理 ===")
        doc_info, chunk_info = process_document_hierarchically(document)
        print(f"文档级关键信息:")
        print(doc_info[:200] + "...")
        print(f"\n块级关键信息 (前3块):")
        for i, info in chunk_info[:3]:
            print(f"\n块 {i+1}:")
            print(info[:100] + "...")
    
    # 清理临时文件
    if os.path.exists("sample_long_document.txt"):
        os.remove("sample_long_document.txt")

if __name__ == "__main__":
    test_long_document_optimization()