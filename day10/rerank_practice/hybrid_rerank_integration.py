# BGE Rerank 与混合检索集成示例

from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 本地模型路径
LOCAL_BGE_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1.5"
LOCAL_RERANK_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-reranker-base"

class HybridSearchWithRerank:
    """混合检索与重排序集成类"""
    
    def __init__(self):
        """
        初始化混合检索与重排序系统
        """
        # 加载BGE Embedding模型
        try:
            print(f"尝试从本地加载BGE Embedding模型: {LOCAL_BGE_MODEL_PATH}")
            self.embedding_model = SentenceTransformer(LOCAL_BGE_MODEL_PATH)
            print("BGE Embedding模型加载成功!")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("从Hugging Face下载BGE Embedding模型...")
            self.embedding_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
            print("BGE Embedding模型下载成功!")
        
        # 加载BGE Rerank模型
        try:
            print(f"尝试从本地加载BGE Rerank模型: {LOCAL_RERANK_MODEL_PATH}")
            self.rerank_model = CrossEncoder(LOCAL_RERANK_MODEL_PATH, max_length=512)
            print("BGE Rerank模型加载成功!")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("从Hugging Face下载BGE Rerank模型...")
            self.rerank_model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
            print("BGE Rerank模型下载成功!")
        
        # 文档和相关数据
        self.documents = []
        self.doc_ids = []
        self.doc_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    
    def add_documents(self, documents, doc_ids=None):
        """
        添加文档并构建索引
        
        参数:
            documents: 文档列表
            doc_ids: 文档ID列表，可选
        """
        self.documents.extend(documents)
        
        if doc_ids:
            self.doc_ids.extend(doc_ids)
        else:
            # 自动生成文档ID
            start_id = len(self.doc_ids)
            self.doc_ids.extend([f"doc_{start_id + i}" for i in range(len(documents))])
        
        # 更新向量嵌入
        print("生成文档向量...")
        self.doc_embeddings = self.embedding_model.encode(self.documents)
        
        # 更新TF-IDF矩阵
        print("构建TF-IDF矩阵...")
        # 中文分词
        def tokenize(text):
            return list(jieba.cut(text))
        
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        print(f"添加了 {len(documents)} 个文档，当前文档总数: {len(self.documents)}")
    
    def hybrid_search(self, query, top_k=5, alpha=0.5, fusion_method='linear'):
        """
        执行混合检索
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            alpha: 向量检索权重 (0-1)
            fusion_method: 融合方法 ('linear' 或 'rrf')
            
        返回:
            排序后的文档列表
        """
        if not self.documents:
            print("错误: 没有添加文档，请先调用 add_documents 方法")
            return []
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算向量相似度
        vector_similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # 计算TF-IDF相似度
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # 融合得分
        if fusion_method == 'linear':
            # 线性融合
            fused_scores = alpha * vector_similarities + (1 - alpha) * tfidf_similarities
        elif fusion_method == 'rrf':
            # RRF融合
            fused_scores = self._rrf_fusion(vector_similarities, tfidf_similarities, k=60)
        else:
            raise ValueError("融合方法必须是 'linear' 或 'rrf'")
        
        # 排序
        ranked_indices = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)[:top_k]
        
        # 整理结果
        results = []
        for i, idx in enumerate(ranked_indices):
            results.append({
                "rank": i + 1,
                "doc_id": self.doc_ids[idx],
                "document": self.documents[idx],
                "score": float(fused_scores[idx])
            })
        
        return results
    
    def hybrid_search_with_rerank(self, query, top_k=5, alpha=0.5, fusion_method='linear', rerank_candidates=10):
        """
        执行混合检索并使用BGE Rerank重排序
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            alpha: 向量检索权重 (0-1)
            fusion_method: 融合方法 ('linear' 或 'rrf')
            rerank_candidates: 重排序的候选数量
            
        返回:
            重排序后的文档列表
        """
        # 执行混合检索获取候选结果
        candidates = self.hybrid_search(query, top_k=rerank_candidates, alpha=alpha, fusion_method=fusion_method)
        
        if not candidates:
            return []
        
        # 提取候选文档
        candidate_docs = [c['document'] for c in candidates]
        candidate_ids = [c['doc_id'] for c in candidates]
        
        # 准备重排序输入
        pairs = [[query, doc] for doc in candidate_docs]
        
        # 执行重排序
        print("正在执行重排序...")
        rerank_scores = self.rerank_model.predict(pairs)
        
        # 排序
        rerank_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)[:top_k]
        
        # 整理结果
        results = []
        for i, idx in enumerate(rerank_indices):
            results.append({
                "rank": i + 1,
                "doc_id": candidate_ids[idx],
                "document": candidate_docs[idx],
                "score": float(rerank_scores[idx])
            })
        
        return results
    
    def _rrf_fusion(self, scores1, scores2, k=60):
        """
        实现Reciprocal Rank Fusion (RRF)融合
        
        参数:
            scores1: 第一种方法的得分
            scores2: 第二种方法的得分
            k: 超参数
            
        返回:
            融合后的得分
        """
        # 计算排名
        rank1 = np.argsort(-scores1)  # 从高到低排序的索引
        rank2 = np.argsort(-scores2)
        
        # 构建排名字典
        rank_dict1 = {i: rank for rank, i in enumerate(rank1, 1)}
        rank_dict2 = {i: rank for rank, i in enumerate(rank2, 1)}
        
        # 计算RRF得分
        fused_scores = []
        for i in range(len(scores1)):
            score = 1 / (k + rank_dict1[i]) + 1 / (k + rank_dict2[i])
            fused_scores.append(score)
        
        return fused_scores
    
    def compare_methods(self, query, top_k=5):
        """
        比较不同方法的检索效果
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            
        返回:
            不同方法的结果
        """
        print(f"\n=== 比较不同方法 (查询: {query}) ===")
        
        # 1. 纯向量检索
        print("\n1. 纯向量检索:")
        vector_results = self.hybrid_search(query, top_k=top_k, alpha=1.0)
        for result in vector_results:
            print(f"{result['rank']}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document'][:80]}...")
        
        # 2. 纯TF-IDF检索
        print("\n2. 纯TF-IDF检索:")
        tfidf_results = self.hybrid_search(query, top_k=top_k, alpha=0.0)
        for result in tfidf_results:
            print(f"{result['rank']}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document'][:80]}...")
        
        # 3. 混合检索
        print("\n3. 混合检索:")
        hybrid_results = self.hybrid_search(query, top_k=top_k, alpha=0.5)
        for result in hybrid_results:
            print(f"{result['rank']}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document'][:80]}...")
        
        # 4. 混合检索 + 重排序
        print("\n4. 混合检索 + 重排序:")
        rerank_results = self.hybrid_search_with_rerank(query, top_k=top_k)
        for result in rerank_results:
            print(f"{result['rank']}. 得分: {result['score']:.4f}")
            print(f"   文档: {result['document'][:80]}...")
        
        return {
            "vector": vector_results,
            "tfidf": tfidf_results,
            "hybrid": hybrid_results,
            "hybrid_rerank": rerank_results
        }
    
    def get_document_count(self):
        """
        获取当前文档数量
        
        返回:
            文档数量
        """
        return len(self.documents)
    
    def clear_documents(self):
        """
        清空所有文档
        """
        self.documents = []
        self.doc_ids = []
        self.doc_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        print("文档已清空")

# 示例用法
def main():
    print("=== 混合检索与重排序集成示例 ===")
    
    # 初始化系统
    searcher = HybridSearchWithRerank()
    
    # 示例文档
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。",
        "强化学习是机器学习的一种方法，它通过试错和奖励机制来学习最佳行为策略。",
        "知识图谱是一种结构化的知识表示方法，它通过节点和边来表示实体和它们之间的关系。",
        "智能推荐系统是一种利用人工智能技术为用户推荐个性化内容的系统。",
        "数据挖掘是从大量数据中提取有用信息的过程。",
        "大数据分析是对大规模数据集进行分析的技术。"
    ]
    
    # 文档ID
    doc_ids = ["ai_overview", "machine_learning", "deep_learning", "nlp", 
               "computer_vision", "reinforcement_learning", "knowledge_graph", 
               "recommendation_system", "data_mining", "big_data_analysis"]
    
    # 添加文档
    searcher.add_documents(documents, doc_ids)
    
    # 示例查询
    queries = [
        "人工智能的定义是什么？",
        "机器学习有哪些方法？",
        "自然语言处理的应用"
    ]
    
    # 对每个查询进行比较
    for query in queries:
        searcher.compare_methods(query, top_k=3)
    
    # 测试动态添加文档
    print("\n=== 测试动态添加文档 ===")
    new_documents = [
        "人工智能在医疗领域的应用包括疾病诊断、药物研发和个性化医疗。",
        "人工智能在金融领域的应用包括风险管理、 fraud detection和算法交易。"
    ]
    
    searcher.add_documents(new_documents)
    
    # 测试新查询
    new_query = "人工智能在医疗和金融领域的应用"
    searcher.compare_methods(new_query, top_k=3)
    
    # 测试文档管理
    print(f"\n=== 文档管理测试 ===")
    print(f"当前文档数量: {searcher.get_document_count()}")
    
    # 清空文档
    searcher.clear_documents()
    print(f"清空后文档数量: {searcher.get_document_count()}")

if __name__ == "__main__":
    main()