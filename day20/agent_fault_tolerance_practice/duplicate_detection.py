# 重复检索终止示例

import numpy as np
from sentence_transformers import SentenceTransformer
import string

# 基于相似度的重复检测器
class SimilarityBasedDuplicateDetector:
    """基于相似度的重复检测器"""
    
    def __init__(self, threshold=0.8):
        """初始化检测器"""
        self.model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self.threshold = threshold
        self.query_history = []
    
    def is_duplicate(self, query):
        """检测查询是否重复"""
        if not self.query_history:
            # 第一次查询，不是重复
            self.query_history.append(query)
            return False
        
        # 计算查询与历史查询的相似度
        query_embedding = self.model.encode([query])[0]
        history_embeddings = self.model.encode(self.query_history)
        
        # 计算余弦相似度
        similarities = np.dot(history_embeddings, query_embedding) / (
            np.linalg.norm(history_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 检查是否有相似度超过阈值的历史查询
        max_similarity = np.max(similarities)
        if max_similarity >= self.threshold:
            return True
        
        # 添加到历史查询
        self.query_history.append(query)
        return False
    
    def clear_history(self):
        """清空历史查询"""
        self.query_history = []

# 基于规则的重复检测器
class RuleBasedDuplicateDetector:
    """基于规则的重复检测器"""
    
    def __init__(self):
        """初始化检测器"""
        self.query_history = []
    
    def is_duplicate(self, query):
        """检测查询是否重复"""
        # 规范化查询
        normalized_query = self._normalize_query(query)
        
        # 检查是否与历史查询重复
        for history_query in self.query_history:
            normalized_history = self._normalize_query(history_query)
            if normalized_query == normalized_history:
                return True
        
        # 添加到历史查询
        self.query_history.append(query)
        return False
    
    def _normalize_query(self, query):
        """规范化查询"""
        # 转换为小写
        query = query.lower()
        # 去除标点符号
        query = query.translate(str.maketrans('', '', string.punctuation))
        # 去除多余的空格
        query = ' '.join(query.split())
        return query
    
    def clear_history(self):
        """清空历史查询"""
        self.query_history = []

# 基于上下文的重复检测器
class ContextAwareDuplicateDetector:
    """基于上下文的重复检测器"""
    
    def __init__(self, threshold=0.8):
        """初始化检测器"""
        self.model = SentenceTransformer("BAAI/bge-base-zh-v1.5")
        self.threshold = threshold
        self.conversation_history = []
    
    def is_duplicate(self, query, context=None):
        """检测查询是否重复"""
        if not self.conversation_history:
            # 第一次查询，不是重复
            self.conversation_history.append((query, context))
            return False
        
        # 构建查询的上下文表示
        if context:
            full_query = f"Context: {context}\nQuery: {query}"
        else:
            full_query = query
        
        # 计算与历史查询的相似度
        query_embedding = self.model.encode([full_query])[0]
        
        for history_query, history_context in self.conversation_history:
            if history_context:
                full_history = f"Context: {history_context}\nQuery: {history_query}"
            else:
                full_history = history_query
            
            history_embedding = self.model.encode([full_history])[0]
            
            # 计算余弦相似度
            similarity = np.dot(query_embedding, history_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(history_embedding)
            )
            
            if similarity >= self.threshold:
                return True
        
        # 添加到历史查询
        self.conversation_history.append((query, context))
        return False
    
    def clear_history(self):
        """清空历史查询"""
        self.conversation_history = []

# 测试重复检测
def test_duplicate_detection():
    """测试重复检测"""
    print("=== 测试重复检测 ===")
    
    # 初始化检测器
    similarity_detector = SimilarityBasedDuplicateDetector()
    rule_detector = RuleBasedDuplicateDetector()
    context_detector = ContextAwareDuplicateDetector()
    
    # 测试查询
    test_cases = [
        ("什么是人工智能", None),
        ("人工智能的定义", None),  # 与第一个查询相似
        ("什么是人工智能？", None),  # 与第一个查询相似（带标点）
        ("机器学习的应用", None),
        ("什么是人工智能", "我想了解人工智能的基本概念"),  # 与第一个查询相同但上下文不同
        ("人工智能的定义", "我想了解人工智能的基本概念")  # 与第二个查询相同但上下文不同
    ]
    
    for i, (query, context) in enumerate(test_cases):
        print(f"\n测试 {i+1}: {query}")
        if context:
            print(f"上下文: {context}")
        
        # 基于相似度的检测
        is_duplicate_similarity = similarity_detector.is_duplicate(query)
        print(f"基于相似度的检测: {'重复' if is_duplicate_similarity else '不重复'}")
        
        # 基于规则的检测
        is_duplicate_rule = rule_detector.is_duplicate(query)
        print(f"基于规则的检测: {'重复' if is_duplicate_rule else '不重复'}")
        
        # 基于上下文的检测
        is_duplicate_context = context_detector.is_duplicate(query, context)
        print(f"基于上下文的检测: {'重复' if is_duplicate_context else '不重复'}")

if __name__ == "__main__":
    test_duplicate_detection()