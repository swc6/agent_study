# 基本混合检索实现

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1.5"

class HybridSearch:
    def __init__(self, model_name='BAAI/bge-base-zh-v1.5'):
        """初始化混合搜索器"""
        # 尝试从本地加载模型
        try:
            self.model = SentenceTransformer(LOCAL_MODEL_PATH)
            print(f"从本地加载模型成功: {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试从 Hugging Face 下载...")
            self.model = SentenceTransformer(model_name)
            print("从 Hugging Face 加载模型成功")
        
        # BM25相关
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        # 向量相关
        self.embeddings = None
        # 文档ID映射
        self.doc_ids = []
    
    def add_documents(self, documents, doc_ids=None):
        """添加文档并构建索引
        
        Args:
            documents: 文档文本列表
            doc_ids: 文档ID列表（可选）
        """
        print(f"添加 {len(documents)} 个文档...")
        self.documents.extend(documents)
        
        # 添加文档ID
        if doc_ids is None:
            # 如果没有提供doc_ids，生成默认ID
            new_doc_ids = [f"doc_{len(self.doc_ids) + i}" for i in range(len(documents))]
        else:
            new_doc_ids = doc_ids
        self.doc_ids.extend(new_doc_ids)
        
        # 构建BM25索引
        print("构建BM25索引...")
        tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.tokenized_corpus.extend(tokenized_corpus)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 生成向量嵌入
        print("生成文档向量...")
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        print(f"当前文档总数: {len(self.documents)}")
    
    def search(self, query, top_k=5, alpha=0.5, fusion_method='linear'):
        """混合搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            alpha: BM25权重，范围0-1
            fusion_method: 融合方法 ('linear', 'rrf', 'normalized')
            
        Returns:
            搜索结果列表
        """
        if not self.bm25 or self.embeddings is None:
            print("没有文档，请先调用add_documents添加文档")
            return []
        
        print(f"搜索查询: '{query}'")
        print(f"融合方法: {fusion_method}, 权重alpha: {alpha}")
        
        # BM25检索
        tokenized_query = list(jieba.cut(query))
        print(f"分词结果: {tokenized_query}")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 向量检索
        query_embedding = self.model.encode([query])[0]
        vector_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 融合得分
        if fusion_method == 'linear':
            # 线性融合
            print("使用线性融合策略")
            fused_scores = alpha * bm25_scores + (1 - alpha) * vector_scores
        elif fusion_method == 'rrf':
            # RRF融合
            print("使用RRF融合策略")
            k = 60
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            vector_ranks = np.argsort(vector_scores)[::-1]
            
            fused_scores = np.zeros(len(self.documents))
            for i, doc_idx in enumerate(bm25_ranks):
                fused_scores[doc_idx] += 1 / (k + i + 1)
            for i, doc_idx in enumerate(vector_ranks):
                fused_scores[doc_idx] += 1 / (k + i + 1)
        elif fusion_method == 'normalized':
            # 归一化融合
            print("使用归一化融合策略")
            # 归一化BM25得分
            bm25_min = min(bm25_scores)
            bm25_max = max(bm25_scores)
            if bm25_max > bm25_min:
                normalized_bm25 = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
            else:
                normalized_bm25 = bm25_scores
            
            # 归一化向量得分
            vector_min = min(vector_scores)
            vector_max = max(vector_scores)
            if vector_max > vector_min:
                normalized_vector = (vector_scores - vector_min) / (vector_max - vector_min)
            else:
                normalized_vector = vector_scores
            
            fused_scores = alpha * normalized_bm25 + (1 - alpha) * normalized_vector
        else:
            # 默认使用线性融合
            print("使用线性融合策略（默认）")
            fused_scores = alpha * bm25_scores + (1 - alpha) * vector_scores
        
        # 排序
        ranked_indices = np.argsort(fused_scores)[::-1][:top_k]
        results = []
        
        for i in ranked_indices:
            results.append({
                'document': self.documents[i],
                'doc_id': self.doc_ids[i],
                'score': fused_scores[i],
                'bm25_score': bm25_scores[i],
                'vector_score': vector_scores[i]
            })
        
        return results
    
    def get_document_count(self):
        """获取当前文档数量"""
        return len(self.documents)
    
    def clear_documents(self):
        """清空所有文档"""
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.embeddings = None
        self.doc_ids = []
        print("已清空所有文档")

# 使用示例
def main():
    print("=== 基本混合检索实现 ===")
    
    # 初始化搜索器
    searcher = HybridSearch()
    
    # 添加文档（带自定义文档ID）
    print("\n=== 第一次添加文档 ===")
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中提取有意义的信息。"
    ]
    
    # 自定义文档ID
    doc_ids = ["ai_001", "ml_001", "dl_001", "nlp_001", "cv_001"]
    searcher.add_documents(documents, doc_ids)
    
    # 显示文档数量
    print(f"\n当前文档数量: {searcher.get_document_count()}")
    
    # 第一次搜索
    print("\n=== 第一次搜索 ===")
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习的关系"
    ]
    
    for query in queries:
        print("\n" + "=" * 70)
        print(f"查询: '{query}'")
        print("=" * 70)
        
        # 使用线性融合
        results = searcher.search(query, top_k=3, alpha=0.5, fusion_method='linear')
        for i, result in enumerate(results):
            print(f"{i+1}. 综合得分: {result['score']:.4f}")
            print(f"   BM25得分: {result['bm25_score']:.4f}")
            print(f"   向量得分: {result['vector_score']:.4f}")
            print(f"   文档ID: {result['doc_id']}")
            print(f"   文档: {result['document']}")
    
    # 添加更多文档
    print("\n=== 添加更多文档 ===")
    more_documents = [
        "强化学习是机器学习的一个领域，它通过与环境交互来学习最优策略。",
        "迁移学习是机器学习的一种方法，它将从一个任务中学到的知识应用到另一个相关任务中。",
        "生成式AI是人工智能的一个领域，它能够生成新的内容，如文本、图像、音频等。"
    ]
    
    # 自定义文档ID
    more_doc_ids = ["rl_001", "tl_001", "genai_001"]
    searcher.add_documents(more_documents, more_doc_ids)
    
    # 显示文档数量
    print(f"\n当前文档数量: {searcher.get_document_count()}")
    
    # 第二次搜索（不需要重新添加文档）
    print("\n=== 第二次搜索（包含新文档） ===")
    new_queries = [
        "强化学习的应用",
        "生成式AI的特点"
    ]
    
    for query in new_queries:
        print("\n" + "=" * 70)
        print(f"查询: '{query}'")
        print("=" * 70)
        
        # 测试不同融合方法
        fusion_methods = ['linear', 'rrf', 'normalized']
        for method in fusion_methods:
            print(f"\n融合方法: {method}")
            print("-" * 50)
            results = searcher.search(query, top_k=3, alpha=0.5, fusion_method=method)
            
            for i, result in enumerate(results):
                print(f"{i+1}. 综合得分: {result['score']:.4f}")
                print(f"   BM25得分: {result['bm25_score']:.4f}")
                print(f"   向量得分: {result['vector_score']:.4f}")
                print(f"   文档ID: {result['doc_id']}")
                print(f"   文档: {result['document']}")
    
    # 清空文档
    print("\n=== 清空文档 ===")
    searcher.clear_documents()
    print(f"当前文档数量: {searcher.get_document_count()}")
    
    # 尝试搜索（应该提示没有文档）
    print("\n=== 尝试搜索（无文档） ===")
    results = searcher.search("人工智能")
    print(f"搜索结果: {results}")

if __name__ == "__main__":
    main()