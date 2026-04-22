# 纯向量检索示例

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 本地模型路径
LOCAL_MODEL_PATH = r"C:\Users\Administrator\.cache\modelscope\hub\models\BAAI\bge-base-zh-v1___5"

class VectorSearch:
    def __init__(self, model_name='BAAI/bge-base-zh-v1.5'):
        """初始化向量搜索器"""
        # 尝试从本地加载模型
        try:
            self.model = SentenceTransformer(LOCAL_MODEL_PATH)
            print(f"从本地加载模型成功: {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试从 Hugging Face 下载...")
            self.model = SentenceTransformer(model_name)
            print("从 Hugging Face 加载模型成功")
        
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents):
        """添加文档并生成嵌入"""
        print(f"添加 {len(documents)} 个文档...")
        self.documents.extend(documents)
        
        # 生成嵌入
        print("生成文档向量...")
        new_embeddings = self.model.encode(documents)
        
        # 合并嵌入
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        print(f"当前文档总数: {len(self.documents)}")
    
    def search(self, query, top_k=5):
        """搜索相关文档"""
        if len(self.documents) == 0:
            return []
        
        # 生成查询向量
        print(f"搜索查询: '{query}'")
        query_embedding = self.model.encode([query])[0]
        
        # 计算相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 排序
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for i in ranked_indices:
            results.append({
                'document': self.documents[i],
                'similarity': similarities[i]
            })
        
        return results

# 使用示例
def main():
    print("=== 纯向量检索示例 ===")
    
    # 初始化搜索器
    searcher = VectorSearch()
    
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
            print(f"{i+1}. 相似度: {result['similarity']:.4f}")
            print(f"   文档: {result['document']}")

if __name__ == "__main__":
    main()