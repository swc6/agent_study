# 高级对话记忆实现

from langchain_community.llms.dashscope import DashScope
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化LLM
llm = DashScope(
    model="qwen-plus",
    temperature=0.7
)

# 初始化窗口记忆（只保留最近的对话）
window_memory = ConversationBufferWindowMemory(k=3)

# 初始化摘要记忆（生成对话摘要）
summary_memory = ConversationSummaryMemory(llm=llm)

# 测试窗口记忆
def test_window_memory():
    print("=== 测试窗口记忆 ===")
    conversation = ConversationChain(
        llm=llm,
        memory=window_memory,
        verbose=True
    )
    
    # 多轮对话
    print("\n轮次 1:")
    response1 = conversation.invoke("你好，我是李四。")
    print(f"助手: {response1['response']}")
    
    print("\n轮次 2:")
    response2 = conversation.invoke("我想学习编程。")
    print(f"助手: {response2['response']}")
    
    print("\n轮次 3:")
    response3 = conversation.invoke("推荐什么语言？")
    print(f"助手: {response3['response']}")
    
    print("\n轮次 4:")
    response4 = conversation.invoke("为什么推荐Python？")
    print(f"助手: {response4['response']}")
    
    print("\n轮次 5:")
    response5 = conversation.invoke("你能记住我的名字吗？")
    print(f"助手: {response5['response']}")
    
    # 查看记忆内容
    print("\n记忆内容:")
    print(window_memory.buffer)

# 测试摘要记忆
def test_summary_memory():
    print("\n=== 测试摘要记忆 ===")
    conversation = ConversationChain(
        llm=llm,
        memory=summary_memory,
        verbose=True
    )
    
    # 多轮对话
    print("\n轮次 1:")
    response1 = conversation.invoke("你好，我是王五。")
    print(f"助手: {response1['response']}")
    
    print("\n轮次 2:")
    response2 = conversation.invoke("我想了解机器学习。")
    print(f"助手: {response2['response']}")
    
    print("\n轮次 3:")
    response3 = conversation.invoke("它有哪些算法？")
    print(f"助手: {response3['response']}")
    
    print("\n轮次 4:")
    response4 = conversation.invoke("监督学习和无监督学习有什么区别？")
    print(f"助手: {response4['response']}")
    
    print("\n轮次 5:")
    response5 = conversation.invoke("你能记住我的名字吗？")
    print(f"助手: {response5['response']}")
    
    # 打印记忆摘要
    print("\n记忆摘要:")
    print(summary_memory.buffer)

if __name__ == "__main__":
    test_window_memory()
    test_summary_memory()