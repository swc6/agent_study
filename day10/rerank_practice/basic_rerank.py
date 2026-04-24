# BGE Rerank 基本使用示例

from sentence_transformers import CrossEncoder
import numpy as np

# 本地模型路径（如果已下载）
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-reranker-base"

class BGERerank:
    """BGE Rerank 重排序类"""
    
    def __init__(self, model_name='BAAI/bge-reranker-base', max_length=512):
        """
        初始化BGE Rerank模型
        
        参数:
            model_name: 模型名称或本地路径
            max_length: 模型最大输入长度
        """
        # 尝试从本地路径加载，如果不存在则从Hugging Face下载
        try:
            print(f"尝试从本地加载模型: {LOCAL_MODEL_PATH}")
            self.model = CrossEncoder(LOCAL_MODEL_PATH, max_length=max_length)
            print("本地模型加载成功!")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print(f"从Hugging Face下载模型: {model_name}")
            self.model = CrossEncoder(model_name, max_length=max_length)
            print("模型下载成功!")
        
        self.max_length = max_length
        self.documents = []
        self.doc_ids = []
    
    def add_documents(self, documents, doc_ids=None):
        """
        添加文档
        
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
        
        print(f"添加了 {len(documents)} 个文档，当前文档总数: {len(self.documents)}")
    
    def rerank(self, query, top_k=5):
        """
        对文档进行重排序
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            
        返回:
            排序后的文档列表，每个元素包含文档内容、文档ID和得分
        """
        if not self.documents:
            print("错误: 没有添加文档，请先调用 add_documents 方法")
            return []
        
        # 准备输入对
        pairs = [[query, doc] for doc in self.documents]
        
        # 计算相关性得分
        print("正在计算相关性得分...")
        scores = self.model.predict(pairs)
        
        # 排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 整理结果
        results = []
        for i, idx in enumerate(ranked_indices):
            results.append({
                "rank": i + 1,
                "doc_id": self.doc_ids[idx],
                "document": self.documents[idx],
                "score": float(scores[idx])
            })
        
        return results
    
    def batch_rerank(self, query, top_k=5, batch_size=32):
        """
        批量处理重排序，适合大量文档
        
        参数:
            query: 查询文本
            top_k: 返回前k个结果
            batch_size: 批处理大小
            
        返回:
            排序后的文档列表
        """
        if not self.documents:
            print("错误: 没有添加文档，请先调用 add_documents 方法")
            return []
        
        scores = []
        
        # 批量处理
        for i in range(0, len(self.documents), batch_size):
            end_idx = min(i + batch_size, len(self.documents))
            batch_docs = self.documents[i:end_idx]
            pairs = [[query, doc] for doc in batch_docs]
            
            # 计算得分
            batch_scores = self.model.predict(pairs)
            scores.extend(batch_scores)
        
        # 排序
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # 整理结果
        results = []
        for i, idx in enumerate(ranked_indices):
            results.append({
                "rank": i + 1,
                "doc_id": self.doc_ids[idx],
                "document": self.documents[idx],
                "score": float(scores[idx])
            })
        
        return results
    
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
        print("文档已清空")

# 示例用法
def main():
    print("=== BGE Rerank 基本使用示例 ===")
    
    # 初始化重排序器
    reranker = BGERerank()
    
    # 示例文档
    documents = [
        "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
        "计算机视觉是人工智能的一个领域，它使计算机能够理解和解释图像和视频。",
        "强化学习是机器学习的一种方法，它通过试错和奖励机制来学习最佳行为策略。",
        "知识图谱是一种结构化的知识表示方法，它通过节点和边来表示实体和它们之间的关系。",
        "智能推荐系统是一种利用人工智能技术为用户推荐个性化内容的系统。"
    ]
    
    # 文档ID
    doc_ids = ["ai_overview", "machine_learning", "deep_learning", "nlp", 
               "computer_vision", "reinforcement_learning", "knowledge_graph", "recommendation_system"]
    
    # 添加文档
    reranker.add_documents(documents, doc_ids)
    
    # 示例查询
    queries = [
        "什么是人工智能？",
        "机器学习的方法有哪些？",
        "自然语言处理的应用场景"
    ]
    
    # 对每个查询进行重排序
    for i, query in enumerate(queries):
        print(f"\n=== 查询 {i+1}: {query} ===")
        
        # 普通重排序
        results = reranker.rerank(query, top_k=3)
        
        # 打印结果
        print("重排序结果:")
        for result in results:
            print(f"{result['rank']}. 得分: {result['score']:.4f}")
            print(f"   ID: {result['doc_id']}")
            print(f"   文档: {result['document'][:100]}...")
    
    # 测试批量处理
    print("\n=== 测试批量处理 ===")
    # 添加更多文档
    more_documents = [
        "数据挖掘是从大量数据中提取有用信息的过程。",
        "大数据分析是对大规模数据集进行分析的技术。",
        "云计算是通过网络提供计算资源的服务模式。",
        "物联网是通过互联网连接各种设备的网络。",
        "区块链是一种分布式账本技术，用于记录交易。"
    ]
    
    reranker.add_documents(more_documents)
    
    # 使用批量处理
    query = "人工智能的相关技术"
    batch_results = reranker.batch_rerank(query, top_k=5, batch_size=4)
    
    print(f"\n批量处理结果 (查询: {query}):")
    for result in batch_results:
        print(f"{result['rank']}. 得分: {result['score']:.4f}")
        print(f"   ID: {result['doc_id']}")
        print(f"   文档: {result['document'][:100]}...")
    
    # 测试文档管理
    print(f"\n=== 文档管理测试 ===")
    print(f"当前文档数量: {reranker.get_document_count()}")
    
    # 清空文档
    reranker.clear_documents()
    print(f"清空后文档数量: {reranker.get_document_count()}")

if __name__ == "__main__":
    main()