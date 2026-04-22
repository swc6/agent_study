# LangChain RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

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