# 综合示例：Agent 反思机制、重复检索终止与异常容错

import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# 初始化嵌入模型
embedding_model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

# 模拟检索函数
def simulate_search(query):
    """模拟检索函数"""
    # 模拟检索时间
    time.sleep(0.5)
    
    # 模拟检索结果
    if "人工智能" in query:
        return ["人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
                "人工智能的发展可以追溯到20世纪50年代。"]
    elif "机器学习" in query:
        return ["机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。"]
    elif "深度学习" in query:
        return ["深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的学习过程。"]
    else:
        return []

# 基于相似度的重复检测器
class DuplicateDetector:
    """重复检测器"""
    
    def __init__(self, threshold=0.8):
        """初始化检测器"""
        self.model = embedding_model
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

# 反思器
class Reflector:
    """反思器"""
    
    def __init__(self):
        """初始化反思器"""
        self.history = []
    
    def add_history(self, query, result, time_taken):
        """添加历史记录"""
        self.history.append({
            "query": query,
            "result": result,
            "time": time_taken
        })
    
    def reflect(self):
        """反思"""
        if not self.history:
            return "继续当前策略"
        
        # 分析历史记录
        num_searches = len(self.history)
        consecutive_failures = 0
        for h in reversed(self.history):
            if not h["result"]:
                consecutive_failures += 1
            else:
                break
        
        # 检查是否有重复结果
        has_duplicate_results = False
        if num_searches >= 2:
            has_duplicate_results = self.history[-1]["result"] == self.history[-2]["result"]
        
        # 检查平均搜索时间
        avg_time = np.mean([h["time"] for h in self.history])
        
        # 制定策略
        if consecutive_failures >= 3:
            return "终止检索，尝试其他方法或向用户请求更多信息"
        elif has_duplicate_results:
            return "调整检索策略，尝试不同的查询词或检索方法"
        elif avg_time > 5:  # 平均搜索时间超过5秒
            return "优化检索策略，减少检索时间"
        elif not self.history[-1]["result"]:
            return "调整查询词，使用更通用或更具体的词汇"
        else:
            return "继续当前策略"
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []

# 异常处理器
class ExceptionHandler:
    """异常处理器"""
    
    @staticmethod
    def handle_exception(func, *args, **kwargs):
        """处理异常"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"执行失败: {str(e)}")
            # 根据异常类型采取不同的处理策略
            if isinstance(e, ConnectionError):
                return "连接失败，请检查网络连接"
            elif isinstance(e, TimeoutError):
                return "请求超时，请稍后重试"
            else:
                return "发生未知错误，请稍后重试"

# 重试装饰器
def retry_on_failure(max_retries=3, delay=1):
    """重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    print(f"执行失败，{delay}秒后重试 ({retries}/{max_retries})...")
                    time.sleep(delay)
        return wrapper
    return decorator

# Agent 类
class Agent:
    """Agent 类"""
    
    def __init__(self):
        """初始化 Agent"""
        self.duplicate_detector = DuplicateDetector()
        self.reflector = Reflector()
        self.exception_handler = ExceptionHandler()
    
    @retry_on_failure(max_retries=3, delay=2)
    def search(self, query):
        """搜索"""
        # 检测是否为重复查询
        if self.duplicate_detector.is_duplicate(query):
            print("检测到重复查询，终止检索")
            return "检测到重复查询，已终止检索"
        
        # 执行搜索
        start_time = time.time()
        result = simulate_search(query)
        time_taken = time.time() - start_time
        
        # 添加到历史记录
        self.reflector.add_history(query, result, time_taken)
        
        # 反思
        reflection = self.reflector.reflect()
        print(f"反思结果: {reflection}")
        
        return result
    
    def process_query(self, query):
        """处理查询"""
        # 安全执行搜索
        result = self.exception_handler.handle_exception(self.search, query)
        return result
    
    def clear_history(self):
        """清空历史记录"""
        self.duplicate_detector.clear_history()
        self.reflector.clear_history()

# 测试综合示例
def test_agent():
    """测试 Agent"""
    print("=== 测试 Agent ===")
    
    # 创建 Agent
    agent = Agent()
    
    # 测试查询
    queries = [
        "什么是人工智能",
        "人工智能的定义",  # 与第一个查询相似
        "机器学习的应用",
        "深度学习的原理",
        "什么是人工智能"  # 重复查询
    ]
    
    for query in queries:
        print(f"\n处理查询: {query}")
        result = agent.process_query(query)
        print(f"查询结果: {result}")

if __name__ == "__main__":
    test_agent()