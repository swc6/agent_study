# 语义切片（Semantic Chunking）
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