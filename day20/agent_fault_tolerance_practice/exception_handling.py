# 异常容错示例

import time
import random

# 异常捕获与处理
def safe_execute(func, *args, **kwargs):
    """安全执行函数，捕获并处理异常"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"执行函数时发生异常: {str(e)}")
        # 根据异常类型采取不同的处理策略
        if isinstance(e, FileNotFoundError):
            return "文件不存在，请检查文件路径"
        elif isinstance(e, ConnectionError):
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

# 故障转移管理器
class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self, primary_service, backup_services):
        """初始化故障转移管理器"""
        self.primary_service = primary_service
        self.backup_services = backup_services
        self.current_service = primary_service
    
    def execute(self, task, *args, **kwargs):
        """执行任务，失败时进行故障转移"""
        try:
            # 尝试使用当前服务执行任务
            return self.current_service.execute(task, *args, **kwargs)
        except Exception as e:
            print(f"当前服务执行失败: {str(e)}")
            
            # 尝试使用备用服务
            for backup_service in self.backup_services:
                try:
                    print(f"尝试使用备用服务: {backup_service.name}")
                    result = backup_service.execute(task, *args, **kwargs)
                    # 切换到备用服务
                    self.current_service = backup_service
                    print(f"已切换到备用服务: {backup_service.name}")
                    return result
                except Exception as e2:
                    print(f"备用服务执行失败: {str(e2)}")
            
            # 所有服务都失败
            raise Exception("所有服务都执行失败")

# 服务类
class Service:
    """服务类"""
    
    def __init__(self, name, reliability):
        """初始化服务"""
        self.name = name
        self.reliability = reliability  # 服务可靠性，0-1之间
    
    def execute(self, task, *args, **kwargs):
        """执行任务"""
        if random.random() > self.reliability:
            raise Exception("服务执行失败")
        return f"{self.name} 执行 {task} 成功"

# 模拟可能抛出异常的函数
def risky_function():
    """可能会抛出异常的函数"""
    if random.random() < 0.7:
        raise ConnectionError("连接失败")
    return "执行成功"

# 模拟文件操作函数
def file_operation():
    """模拟文件操作"""
    if random.random() < 0.5:
        raise FileNotFoundError("文件不存在")
    return "文件操作成功"

# 测试异常容错
def test_exception_handling():
    """测试异常容错"""
    print("=== 测试异常容错 ===")
    
    # 测试安全执行
    print("\n测试安全执行:")
    result1 = safe_execute(risky_function)
    print(f"执行结果: {result1}")
    
    result2 = safe_execute(file_operation)
    print(f"执行结果: {result2}")
    
    # 测试重试机制
    print("\n测试重试机制:")
    @retry_on_failure(max_retries=3, delay=2)
    def unreliable_function():
        """可能会失败的函数"""
        if random.random() < 0.7:
            raise ConnectionError("连接失败")
        return "执行成功"
    
    try:
        result3 = unreliable_function()
        print(f"执行结果: {result3}")
    except Exception as e:
        print(f"最终执行失败: {str(e)}")
    
    # 测试故障转移
    print("\n测试故障转移:")
    # 创建服务
    primary_service = Service("主服务", 0.3)  # 可靠性较低
    backup_service1 = Service("备用服务1", 0.7)
    backup_service2 = Service("备用服务2", 0.9)
    
    # 创建故障转移管理器
    failover_manager = FailoverManager(primary_service, [backup_service1, backup_service2])
    
    # 执行任务
    try:
        result4 = failover_manager.execute("检索任务")
        print(f"执行结果: {result4}")
    except Exception as e:
        print(f"执行失败: {str(e)}")

if __name__ == "__main__":
    test_exception_handling()