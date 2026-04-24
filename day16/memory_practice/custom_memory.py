# 自定义对话记忆实现

from langchain_community.llms.dashscope import DashScope
from langchain.memory import BaseMemory
from langchain.chains import ConversationChain
from pydantic import BaseModel
from typing import Dict, List, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 自定义记忆类
class CustomMemory(BaseMemory, BaseModel):
    """自定义记忆类"""
    
    # 对话历史
    conversation_history: List[Dict[str, str]] = []
    # 用户信息
    user_info: Dict[str, str] = {}
    # 最大历史长度
    max_history_length: int = 5
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量"""
        return ["history", "user_info"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量"""
        # 构建对话历史
        history = ""
        for item in self.conversation_history:
            history += f"用户: {item['user']}\n"
            history += f"助手: {item['assistant']}\n"
        
        # 构建用户信息
        user_info_str = ""
        if self.user_info:
            user_info_str = "用户信息: " + ", ".join([f"{k}: {v}" for k, v in self.user_info.items()])
        
        return {
            "history": history,
            "user_info": user_info_str
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存上下文"""
        # 添加对话历史
        self.conversation_history.append({
            "user": inputs["input"],
            "assistant": outputs["response"]
        })
        
        # 保持历史长度
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear(self) -> None:
        """清空记忆"""
        self.conversation_history = []
        self.user_info = {}

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化自定义记忆
custom_memory = CustomMemory()

# 设置用户信息
custom_memory.user_info = {
    "name": "赵六",
    "interest": "人工智能",
    "level": "初学者"
}

# 自定义提示词
custom_prompt = """
你是一个智能助手，需要根据对话历史和用户信息来回答问题。

用户信息:
{user_info}

对话历史:
{history}

用户: {input}
助手:
"""

# 初始化对话链
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(input_variables=["history", "user_info", "input"], template=custom_prompt)

conversation = ConversationChain(
    llm=llm,
    memory=custom_memory,
    prompt=prompt,
    verbose=True
)

# 测试自定义记忆
def test_custom_memory():
    print("=== 测试自定义记忆 ===")
    
    # 多轮对话
    print("\n轮次 1:")
    response1 = conversation.invoke("你好！")
    print(f"助手: {response1['response']}")
    
    print("\n轮次 2:")
    response2 = conversation.invoke("我想学习人工智能，有什么建议吗？")
    print(f"助手: {response2['response']}")
    
    print("\n轮次 3:")
    response3 = conversation.invoke("推荐哪些学习资源？")
    print(f"助手: {response3['response']}")
    
    print("\n轮次 4:")
    response4 = conversation.invoke("需要学习哪些数学知识？")
    print(f"助手: {response4['response']}")
    
    print("\n轮次 5:")
    response5 = conversation.invoke("你能记住我的名字吗？")
    print(f"助手: {response5['response']}")
    
    # 查看记忆内容
    print("\n记忆内容:")
    print(f"用户信息: {custom_memory.user_info}")
    print("对话历史:")
    for item in custom_memory.conversation_history:
        print(f"  用户: {item['user']}")
        print(f"  助手: {item['assistant']}")

if __name__ == "__main__":
    test_custom_memory()