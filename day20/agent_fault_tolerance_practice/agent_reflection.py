# Agent 反思机制示例

import numpy as np
import time

# 基于规则的反思器
class RuleBasedReflector:
    """基于规则的反思器"""
    
    def __init__(self):
        """初始化反思器"""
        self.rules = [
            # 规则：如果连续两次检索结果相同，考虑调整检索策略
            {"condition": lambda history: len(history) >= 2 and history[-1]["result"] == history[-2]["result"],
             "action": "调整检索策略，尝试不同的查询词或检索方法"},
            # 规则：如果检索结果为空，考虑调整查询词
            {"condition": lambda history: len(history) > 0 and not history[-1]["result"],
             "action": "调整查询词，使用更通用或更具体的词汇"},
            # 规则：如果连续多次检索都失败，考虑终止检索
            {"condition": lambda history: len(history) >= 3 and all(not h["result"] for h in history[-3:]),
             "action": "终止检索，尝试其他方法或向用户请求更多信息"},
            # 规则：如果检索时间过长，考虑优化检索策略
            {"condition": lambda history: len(history) > 0 and history[-1]["time"] > 5,
             "action": "优化检索策略，减少检索时间"}
        ]
    
    def reflect(self, history):
        """根据历史行为进行反思"""
        for rule in self.rules:
            if rule["condition"](history):
                return rule["action"]
        return "继续当前策略"

# 基于自我评估的反思器
class SelfEvaluatingReflector:
    """基于自我评估的反思器"""
    
    def __init__(self, task_goal):
        """初始化反思器"""
        self.task_goal = task_goal
        self.evaluation_metrics = {
            "relevance": self._evaluate_relevance,
            "completeness": self._evaluate_completeness,
            "efficiency": self._evaluate_efficiency
        }
    
    def _evaluate_relevance(self, history, current_result):
        """评估结果的相关性"""
        # 简单的相关性评估：检查结果是否包含任务目标中的关键词
        goal_keywords = self.task_goal.split()
        result_text = " ".join([item for item in current_result])
        matching_keywords = [kw for kw in goal_keywords if kw in result_text]
        return len(matching_keywords) / len(goal_keywords) if goal_keywords else 0
    
    def _evaluate_completeness(self, history, current_result):
        """评估结果的完整性"""
        # 简单的完整性评估：检查结果的数量和长度
        if not current_result:
            return 0
        avg_length = np.mean([len(item) for item in current_result])
        return min(1, (len(current_result) * avg_length) / 1000)  # 假设1000个字符为完整
    
    def _evaluate_efficiency(self, history):
        """评估执行效率"""
        # 简单的效率评估：检查执行时间和检索次数
        if not history:
            return 1
        total_time = sum([h["time"] for h in history])
        num_searches = len(history)
        # 时间越短，检索次数越少，效率越高
        return max(0, 1 - (total_time / 60 + num_searches / 10) / 2)
    
    def reflect(self, history, current_result):
        """根据历史行为和当前结果进行反思"""
        if not history:
            return "继续当前策略"
        
        # 评估当前结果
        evaluations = {}
        for metric_name, metric_func in self.evaluation_metrics.items():
            if metric_name == "efficiency":
                evaluations[metric_name] = metric_func(history)
            else:
                evaluations[metric_name] = metric_func(history, current_result)
        
        # 计算综合评分
        scores = list(evaluations.values())
        avg_score = np.mean(scores)
        
        # 根据评分决定行动
        if avg_score >= 0.8:
            return "继续当前策略"
        elif avg_score >= 0.5:
            return "调整当前策略"
        elif avg_score >= 0.2:
            return "尝试不同的策略"
        else:
            return "终止当前策略，向用户请求更多信息"

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

# 测试反思机制
def test_reflection():
    """测试反思机制"""
    print("=== 测试反思机制 ===")
    
    # 初始化反思器
    rule_reflector = RuleBasedReflector()
    self_reflector = SelfEvaluatingReflector("了解人工智能和机器学习")
    
    # 模拟历史记录
    history = []
    
    # 测试查询
    queries = [
        "什么是人工智能",
        "人工智能的定义",  # 与第一个查询相似
        "机器学习的定义",
        "深度学习的定义",
        "什么是人工智能"  # 重复查询
    ]
    
    for i, query in enumerate(queries):
        print(f"\n查询 {i+1}: {query}")
        
        # 模拟检索
        start_time = time.time()
        result = simulate_search(query)
        time_taken = time.time() - start_time
        
        # 添加到历史记录
        history.append({
            "query": query,
            "result": result,
            "time": time_taken
        })
        
        # 打印结果
        print(f"检索结果: {result}")
        print(f"检索时间: {time_taken:.2f} 秒")
        
        # 基于规则的反思
        rule_reflection = rule_reflector.reflect(history)
        print(f"基于规则的反思: {rule_reflection}")
        
        # 基于自我评估的反思
        self_reflection = self_reflector.reflect(history, result)
        print(f"基于自我评估的反思: {self_reflection}")

if __name__ == "__main__":
    test_reflection()