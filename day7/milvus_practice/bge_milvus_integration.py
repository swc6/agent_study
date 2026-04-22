# BGE与Milvus文档管理集成示例

from milvus_document_manager import MilvusDocumentManager
from sentence_transformers import SentenceTransformer
import numpy as np

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1_5"

class BGE_Milvus_Integration:
    """BGE与Milvus集成类"""
    
    def __init__(self, collection_name="bge_documents"):
        """初始化集成类
        
        Args:
            collection_name: 集合名称
        """
        # 初始化文档管理器
        self.manager = MilvusDocumentManager(collection_name=collection_name)
        
        # 加载BGE模型
        self.load_model()
        
        # 连接到Milvus并创建集合
        self.setup_milvus()
    
    def load_model(self):
        """加载BGE模型"""
        try:
            # 尝试从本地加载
            self.model = SentenceTransformer(LOCAL_MODEL_PATH)
            print(f"从本地加载模型成功: {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试从 Hugging Face 下载...")
            self.model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
            print("从 Hugging Face 加载模型成功")
    
    def setup_milvus(self):
        """设置Milvus"""
        # 连接到Milvus
        if not self.manager.connect():
            raise Exception("无法连接到Milvus服务")
        
        # 创建集合
        if not self.manager.create_collection():
            raise Exception("无法创建Milvus集合")
    
    def add_documents(self, documents, doc_ids=None):
        """添加文档到Milvus
        
        Args:
            documents: 文档文本列表
            doc_ids: 文档ID列表（可选）
        """
        # 生成向量
        print("生成文档向量...")
        embeddings = self.model.encode(documents)
        embeddings = np.array(embeddings).astype(np.float32)
        
        # 插入文档
        return self.manager.insert_documents(documents, embeddings, doc_ids)
    
    def search(self, query, top_k=5):
        """搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        # 生成查询向量
        print(f"搜索查询: '{query}'")
        query_embedding = self.model.encode([query])[0]
        
        # 执行搜索
        results = self.manager.search(query_embedding, top_k=top_k)
        
        # 处理搜索结果
        if results:
            processed_results = []
            for i, hit in enumerate(results[0]):
                similarity = 1 / (1 + hit.distance)  # 转换为相似度
                processed_results.append({
                    'rank': i + 1,
                    'document': hit.entity.get('text'),
                    'doc_id': hit.entity.get('doc_id'),
                    'distance': hit.distance,
                    'similarity': similarity
                })
            return processed_results
        else:
            return []
    
    def close(self):
        """关闭资源"""
        # 释放集合
        self.manager.release_collection()
        
        # 断开连接
        self.manager.disconnect()

# 使用示例
def main():
    print("=== BGE与Milvus集成示例 ===")
    
    # 初始化集成类
    integration = BGE_Milvus_Integration()
    
    try:
        # 添加文档
        print("\n添加示例文档...")
        documents = [
            "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
            "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。",
            "深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。",
            "自然语言处理是人工智能的一个领域，它使计算机能够理解、解释和生成人类语言。",
            "计算机视觉是人工智能的一个领域，它使计算机能够从图像或视频中提取有意义的信息。"
        ]
        
        # 手动插入文档
        integration.add_documents(documents)
        
        # 多次搜索，验证不需要重复插入
        queries = [
            "什么是人工智能？",
            "机器学习和深度学习的关系",
            "自然语言处理的应用"
        ]
        
        for query in queries:
            print("\n" + "-" * 50)
            results = integration.search(query, top_k=3)
            
            print(f"查询: '{query}'")
            for result in results:
                print(f"{result['rank']}. 相似度: {result['similarity']:.4f}")
                print(f"   文档: {result['document']}")
                print(f"   文档ID: {result['doc_id']}")
        
        # 演示添加新文档
        print("\n" + "=" * 70)
        print("添加新文档...")
        new_documents = [
            "强化学习是机器学习的一个领域，它通过与环境交互来学习最优策略。",
            "迁移学习是机器学习的一种方法，它将从一个任务中学到的知识应用到另一个相关任务中。"
        ]
        
        # 手动插入新文档
        integration.add_documents(new_documents)
        
        # 再次搜索，验证新文档已被添加
        print("\n搜索包含新文档的结果:")
        results = integration.search("强化学习", top_k=2)
        for result in results:
            print(f"{result['rank']}. 相似度: {result['similarity']:.4f}")
            print(f"   文档: {result['document']}")
            print(f"   文档ID: {result['doc_id']}")
            
    finally:
        # 关闭资源
        integration.close()

if __name__ == "__main__":
    main()