# 基础对话记忆实现

from langchain_community.llms.dashscope import DashScope
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化记忆
memory = ConversationBufferMemory()

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 测试多轮对话
def test_basic_memory():
    print("=== 测试基础对话记忆 ===")
    
    # 第一轮
    print("\n轮次 1:")
    response1 = conversation.invoke("你好，我是张三。")
    print(f"助手: {response1['response']}")
    
    # 第二轮
    print("\n轮次 2:")
    response2 = conversation.invoke("我想了解一下人工智能。")
    print(f"助手: {response2['response']}")
    
    # 第三轮
    print("\n轮次 3:")
    response3 = conversation.invoke("它和机器学习有什么关系？")
    print(f"助手: {response3['response']}")
    
    # 第四轮
    print("\n轮次 4:")
    response4 = conversation.invoke("能举个例子吗？")
    print(f"助手: {response4['response']}")
    
    # 第五轮
    print("\n轮次 5:")
    response5 = conversation.invoke("你能记住我的名字吗？")
    print(f"助手: {response5['response']}")
    
    # 查看记忆内容
    print("\n记忆内容:")
    print(memory.buffer)

if __name__ == "__main__":
    test_basic_memory()