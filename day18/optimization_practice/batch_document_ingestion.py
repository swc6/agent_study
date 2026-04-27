# 批量文档入库示例

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import os
import time

# 加载文档
def load_document(path):
    """加载文档"""
    try:
        loader = TextLoader(path, encoding='utf-8')
        documents = loader.load()
        return documents[0]
    except Exception as e:
        print(f"加载文档 {path} 失败: {str(e)}")
        return None

# 分块文档
def chunk_document(document, chunk_size=1000, chunk_overlap=200):
    """分块文档"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents([document])
    return chunks

# 处理单个文档
def process_single_document(path):
    """处理单个文档"""
    doc = load_document(path)
    if doc:
        chunks = chunk_document(doc)
        return (path, chunks, None)
    else:
        return (path, None, "加载失败")

# 批量处理文档
def batch_process_documents(document_paths, max_workers=4):
    """批量处理文档"""
    all_chunks = []
    errors = []
    start_time = time.time()
    
    # 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_document, path): path for path in document_paths}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            path, chunks, error = future.result()
            if error:
                errors.append((path, error))
            else:
                all_chunks.extend(chunks)
            
            # 打印进度
            processed = len(all_chunks) + len(errors)
            print(f"处理进度: {processed}/{len(document_paths)}")
    
    end_time = time.time()
    print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")
    
    return all_chunks, errors

# 带重试机制的文档处理
def process_with_retry(document_paths, max_retries=3):
    """带重试机制的文档处理"""
    results = []
    errors = []
    start_time = time.time()
    
    for path in document_paths:
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                doc = load_document(path)
                if doc:
                    chunks = chunk_document(doc)
                    results.append((path, chunks))
                    success = True
                else:
                    raise Exception("文档加载失败")
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    errors.append((path, str(e)))
                else:
                    print(f"处理 {path} 失败，重试 {retries}/{max_retries}...")
        
        # 打印进度
        processed = len(results) + len(errors)
        print(f"处理进度: {processed}/{len(document_paths)}")
    
    end_time = time.time()
    print(f"处理完成，耗时: {end_time - start_time:.2f} 秒")
    
    return results, errors

# 测试批量文档入库
def test_batch_document_ingestion():
    """测试批量文档入库"""
    print("=== 测试批量文档入库 ===")
    
    # 创建示例文档
    sample_docs = [
        "# 文档1\n人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "# 文档2\n机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "# 文档3\n深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "# 文档4\n自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "# 文档5\n计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。"
    ]
    
    # 保存示例文档
    doc_paths = []
    for i, content in enumerate(sample_docs):
        path = f"sample_doc_{i+1}.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        doc_paths.append(path)
    
    # 测试批量处理
    print("\n=== 测试并行批量处理 ===")
    chunks, errors = batch_process_documents(doc_paths, max_workers=2)
    
    print(f"处理完成，成功 {len(chunks)} 个文档块，失败 {len(errors)} 个文档")
    
    if errors:
        print("失败的文档:")
        for path, error in errors:
            print(f"- {path}: {error}")
    
    # 测试带重试机制的处理
    print("\n=== 测试带重试机制的处理 ===")
    # 创建一个会失败的文档
    error_doc_path = "error_doc.txt"
    with open(error_doc_path, 'w', encoding='utf-8') as f:
        f.write("# 错误文档")
    
    # 修改文件权限使其不可读（Windows可能需要管理员权限）
    try:
        os.chmod(error_doc_path, 0o000)
    except Exception as e:
        print(f"无法修改文件权限: {str(e)}")
    
    # 测试带重试机制的处理
    test_paths = doc_paths + [error_doc_path]
    results, retry_errors = process_with_retry(test_paths)
    
    print(f"处理完成，成功 {len(results)} 个文档，失败 {len(retry_errors)} 个文档")
    
    if retry_errors:
        print("失败的文档:")
        for path, error in retry_errors:
            print(f"- {path}: {error}")
    
    # 清理临时文件
    for path in doc_paths + [error_doc_path]:
        if os.path.exists(path):
            try:
                # 恢复文件权限
                os.chmod(path, 0o666)
                os.remove(path)
            except Exception as e:
                print(f"无法删除文件 {path}: {str(e)}")

if __name__ == "__main__":
    test_batch_document_ingestion()