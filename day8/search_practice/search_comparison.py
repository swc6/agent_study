# 检索方法对比示例

from vector_search import VectorSearch
from bm25_search import BM25Search

# 评估函数
def evaluate(searcher, queries, relevant_docs, top_k=3):
    """评估检索系统性能"""
    precisions = []
    recalls = []
    
    for i, query in enumerate(queries):
        results = searcher.search(query, top_k=top_k)
        retrieved_indices = [searcher.documents.index(result['document']) for result in results]
        
        # 计算精确率
        relevant_retrieved = set(retrieved_indices) & set(relevant_docs[i])
        precision = len(relevant_retrieved) / len(retrieved_indices) if retrieved_indices else 0
        precisions.append(precision)
        
        # 计算召回率
        recall = len(relevant_retrieved) / len(relevant_docs[i]) if relevant_docs[i] else 0
        recalls.append(recall)
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1
    }

# 主函数
def main():
    print("=== 检索方法对比示例 ===")
    
    # 测试文档
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中提取有意义的信息。",
        "强化学习是机器学习的一个领域，它通过与环境交互来学习最优策略。",
        "迁移学习是机器学习的一种方法，它将从一个任务中学到的知识应用到另一个相关任务中。",
        "生成式AI是人工智能的一个领域，它能够生成新的内容，如文本、图像、音频等。"
    ]
    
    # 测试查询和相关文档
    queries = [
        "人工智能的定义",
        "机器学习的应用",
        "深度学习的原理",
        "自然语言处理的作用",
        "计算机视觉的应用场景"
    ]
    
    # 相关文档索引
    relevant_docs = [
        [0],      # "人工智能的定义" 相关文档
        [1, 2, 5, 6],  # "机器学习的应用" 相关文档
        [2],      # "深度学习的原理" 相关文档
        [3],      # "自然语言处理的作用" 相关文档
        [4]       # "计算机视觉的应用场景" 相关文档
    ]
    
    # 初始化搜索器
    print("\n初始化向量搜索器...")
    vector_searcher = VectorSearch()
    vector_searcher.add_documents(documents)
    
    print("\n初始化BM25搜索器...")
    bm25_searcher = BM25Search()
    bm25_searcher.add_documents(documents)
    
    # 测试查询
    for query in queries:
        print("\n" + "=" * 70)
        print(f"查询: '{query}'")
        print("=" * 70)
        
        # 向量检索结果
        print("\n向量检索结果:")
        vector_results = vector_searcher.search(query, top_k=3)
        for i, result in enumerate(vector_results):
            print(f"{i+1}. 相似度: {result['similarity']:.4f}")
            print(f"   文档: {result['document']}")
        
        # BM25检索结果
        print("\nBM25检索结果:")
        bm25_results = bm25_searcher.search(query, top_k=3)
        for i, result in enumerate(bm25_results):
            print(f"{i+1}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document']}")
    
    # 评估性能
    print("\n" + "=" * 70)
    print("性能评估")
    print("=" * 70)
    
    vector_metrics = evaluate(vector_searcher, queries, relevant_docs)
    print("\n向量检索评估结果:")
    print(f"精确率: {vector_metrics['precision']:.4f}")
    print(f"召回率: {vector_metrics['recall']:.4f}")
    print(f"F1分数: {vector_metrics['f1']:.4f}")
    
    bm25_metrics = evaluate(bm25_searcher, queries, relevant_docs)
    print("\nBM25检索评估结果:")
    print(f"精确率: {bm25_metrics['precision']:.4f}")
    print(f"召回率: {bm25_metrics['recall']:.4f}")
    print(f"F1分数: {bm25_metrics['f1']:.4f}")
    
    # 比较
    print("\n" + "=" * 70)
    print("方法比较")
    print("=" * 70)
    print("向量检索优势: 理解语义，识别同义词，处理模糊查询")
    print("BM25检索优势: 速度快，对关键词敏感，适合精确匹配")

if __name__ == "__main__":
    main()