# 不同切片策略对比
from langchain_text_splitters import RecursiveCharacterTextSplitter

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