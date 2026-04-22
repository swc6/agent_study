# BM25 稀疏检索示例

from rank_bm25 import BM25Okapi
import jieba

class BM25Search:
    def __init__(self):
        """初始化BM25搜索器"""
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
    
    def add_documents(self, documents):
        """添加文档并构建索引"""
        print(f"添加 {len(documents)} 个文档...")
        self.documents.extend(documents)
        
        # 分词
        print("对文档进行分词...")
        new_tokenized = [list(jieba.cut(doc)) for doc in documents]
        self.tokenized_corpus.extend(new_tokenized)
        
        # 重新构建BM25索引
        print("构建BM25索引...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"当前文档总数: {len(self.documents)}")
    
    def search(self, query, top_k=5):
        """搜索相关文档"""
        if not self.bm25:
            return []
        
        # 分词
        print(f"搜索查询: '{query}'")
        tokenized_query = list(jieba.cut(query))
        print(f"分词结果: {tokenized_query}")
        
        # 搜索
        scores = self.bm25.get_scores(tokenized_query)
        
        # 排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        
        for i in ranked_indices:
            results.append({
                'document': self.documents[i],
                'score': scores[i]
            })
        
        return results

# 使用示例
def main():
    print("=== BM25 稀疏检索示例 ===")
    
    # 初始化搜索器
    searcher = BM25Search()
    
    # 添加文档
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
    
    searcher.add_documents(documents)
    
    # 测试查询
    queries = [
        "什么是人工智能？",
        "机器学习的应用",
        "深度学习的原理",
        "自然语言处理的作用",
        "计算机视觉的应用场景"
    ]
    
    for query in queries:
        print("\n" + "-" * 50)
        results = searcher.search(query, top_k=3)
        
        print(f"查询: '{query}'")
        for i, result in enumerate(results):
            print(f"{i+1}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document']}")

if __name__ == "__main__":
    main()