# 带重排序的混合检索实现

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1.5"
LOCAL_RERANK_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-reranker-base"

class EnhancedHybridSearch:
    def __init__(self, model_name='BAAI/bge-base-zh-v1.5', rerank_model='BAAI/bge-reranker-base'):
        """初始化增强的混合搜索器"""
        # 尝试从本地加载嵌入模型
        try:
            self.model = SentenceTransformer(LOCAL_MODEL_PATH)
            print(f"从本地加载嵌入模型成功: {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"本地嵌入模型加载失败: {e}")
            print("尝试从 Hugging Face 下载...")
            self.model = SentenceTransformer(model_name)
            print("从 Hugging Face 加载嵌入模型成功")
        
        # 尝试从本地加载重排序模型
        try:
            self.reranker = CrossEncoder(LOCAL_RERANK_MODEL_PATH)
            print(f"从本地加载重排序模型成功: {LOCAL_RERANK_MODEL_PATH}")
        except Exception as e:
            print(f"本地重排序模型加载失败: {e}")
            print("尝试从 Hugging Face 下载...")
            self.reranker = CrossEncoder(rerank_model)
            print("从 Hugging Face 加载重排序模型成功")
        
        # BM25相关
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        # 向量相关
        self.embeddings = None
    
    def add_documents(self, documents):
        """添加文档并构建索引"""
        print(f"添加 {len(documents)} 个文档...")
        self.documents.extend(documents)
        
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
    
    def search(self, query, top_k=5, alpha=0.5, fusion_method='linear', rerank=True):
        """增强的混合搜索"""
        if not self.bm25 or self.embeddings is None:
            return []
        
        print(f"搜索查询: '{query}'")
        print(f"融合方法: {fusion_method}, 权重alpha: {alpha}, 重排序: {rerank}")
        
        # 基础混合搜索（获取更多结果用于重排）
        candidate_results = self._basic_hybrid_search(query, top_k * 3, alpha, fusion_method)
        
        if not rerank:
            return candidate_results[:top_k]
        
        # 使用Rerank模型重排
        print("使用BGE Rerank模型进行重排序...")
        pairs = [[query, result['document']] for result in candidate_results]
        scores = self.reranker.predict(pairs)
        
        # 按Rerank得分排序
        rerank_results = []
        for i, result in enumerate(candidate_results):
            rerank_results.append({
                'document': result['document'],
                'score': scores[i],
                'bm25_score': result['bm25_score'],
                'vector_score': result['vector_score'],
                'rerank_score': scores[i]
            })
        
        # 排序并返回前top_k结果
        rerank_results.sort(key=lambda x: x['score'], reverse=True)
        return rerank_results[:top_k]
    
    def _basic_hybrid_search(self, query, top_k, alpha, fusion_method):
        """基础混合搜索"""
        # BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 向量检索
        query_embedding = self.model.encode([query])[0]
        vector_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 融合得分
        if fusion_method == 'linear':
            fused_scores = alpha * bm25_scores + (1 - alpha) * vector_scores
        elif fusion_method == 'rrf':
            k = 60
            bm25_ranks = np.argsort(bm25_scores)[::-1]
            vector_ranks = np.argsort(vector_scores)[::-1]
            
            fused_scores = np.zeros(len(self.documents))
            for i, doc_idx in enumerate(bm25_ranks):
                fused_scores[doc_idx] += 1 / (k + i + 1)
            for i, doc_idx in enumerate(vector_ranks):
                fused_scores[doc_idx] += 1 / (k + i + 1)
        elif fusion_method == 'normalized':
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
            fused_scores = alpha * bm25_scores + (1 - alpha) * vector_scores
        
        # 排序
        ranked_indices = np.argsort(fused_scores)[::-1][:top_k]
        results = []
        
        for i in ranked_indices:
            results.append({
                'document': self.documents[i],
                'score': fused_scores[i],
                'bm25_score': bm25_scores[i],
                'vector_score': vector_scores[i]
            })
        
        return results

# 使用示例
def main():
    print("=== 带重排序的混合检索实现 ===")
    
    # 初始化搜索器
    searcher = EnhancedHybridSearch()
    
    # 添加文档
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中提取有意义的信息。",
        "强化学习是机器学习的一个领域，它通过与环境交互来学习最优策略。",
        "迁移学习是机器学习的一种方法，它将从一个任务中学到的知识应用到另一个相关任务中。",
        "生成式AI是人工智能的一个领域，它能够生成新的内容，如文本、图像、音频等。",
        "人工智能的历史可以追溯到1956年的达特茅斯会议，当时科学家们首次提出了人工智能的概念。",
        "未来的人工智能将更加智能化，能够理解复杂的人类情感和意图，为人类提供更加个性化的服务。"
    ]
    
    searcher.add_documents(documents)
    
    # 测试查询
    queries = [
        "什么是人工智能？",
        "机器学习和深度学习的关系",
        "自然语言处理的应用"
    ]
    
    for query in queries:
        print("\n" + "=" * 80)
        print(f"查询: '{query}'")
        print("=" * 80)
        
        # 不带重排序的结果
        print("\n1. 不带重排序的结果:")
        print("-" * 60)
        results_without_rerank = searcher.search(query, top_k=3, alpha=0.5, rerank=False)
        for i, result in enumerate(results_without_rerank):
            print(f"{i+1}. 得分: {result['score']:.4f}")
            print(f"   BM25得分: {result['bm25_score']:.4f}")
            print(f"   向量得分: {result['vector_score']:.4f}")
            print(f"   文档: {result['document']}")
        
        # 带重排序的结果
        print("\n2. 带重排序的结果:")
        print("-" * 60)
        results_with_rerank = searcher.search(query, top_k=3, alpha=0.5, rerank=True)
        for i, result in enumerate(results_with_rerank):
            print(f"{i+1}. 重排序得分: {result['rerank_score']:.4f}")
            print(f"   原始得分: {result['score']:.4f}")
            print(f"   BM25得分: {result['bm25_score']:.4f}")
            print(f"   向量得分: {result['vector_score']:.4f}")
            print(f"   文档: {result['document']}")

if __name__ == "__main__":
    main()