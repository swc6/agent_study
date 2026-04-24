# LangChain 链与模板集成示例

from langchain_community.llms.dashscope import DashScope
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ChainTemplateIntegration:
    """LangChain链与模板集成示例"""
    
    def __init__(self):
        """
        初始化链与模板集成示例
        """
        # 初始化LLM
        self.llm = DashScope(
            model="qwen-plus",
            temperature=0.7
        )
        print("LLM初始化成功！")
        
        # 初始化记忆
        self.memory = ConversationBufferMemory()
    
    def create_qa_chain(self):
        """
        创建问答链
        """
        print("\n=== 创建问答链 ===")
        
        # 创建Prompt模板
        prompt = PromptTemplate(
            input_variables=["question"],
            template="请回答以下问题：{question}"
        )
        
        # 创建LLMChain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        return chain
    
    def create_summarization_chain(self):
        """
        创建摘要链
        """
        print("\n=== 创建摘要链 ===")
        
        # 创建Prompt模板
        prompt = PromptTemplate(
            input_variables=["text"],
            template="请为以下文本生成一个摘要，不超过100字：\n{text}"
        )
        
        # 创建LLMChain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        return chain
    
    def create_translation_chain(self):
        """
        创建翻译链
        """
        print("\n=== 创建翻译链 ===")
        
        # 创建Prompt模板
        prompt = PromptTemplate(
            input_variables=["text", "target_language"],
            template="请将以下文本翻译成{target_language}：\n{text}"
        )
        
        # 创建LLMChain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        return chain
    
    def test_qa_chain(self):
        """
        测试问答链
        """
        print("\n=== 测试问答链 ===")
        
        chain = self.create_qa_chain()
        
        # 测试问题
        questions = [
            "什么是人工智能？",
            "机器学习和深度学习的区别是什么？",
            "人工智能的未来发展趋势是什么？"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = chain.run(question)
            print(f"回答: {result}")
    
    def test_summarization_chain(self):
        """
        测试摘要链
        """
        print("\n=== 测试摘要链 ===")
        
        chain = self.create_summarization_chain()
        
        # 测试文本
        text = "人工智能（Artificial Intelligence，简称AI）是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能的发展可以分为三个阶段：弱人工智能、强人工智能和超人工智能。弱人工智能是指只能完成特定任务的人工智能，如语音识别、图像识别等。强人工智能是指具有人类般的智能，可以理解和学习任何智力任务的人工智能。超人工智能是指超越人类智能的人工智能，具有自我意识和创造力。人工智能的应用领域非常广泛，包括医疗、金融、教育、交通、娱乐等。未来，人工智能将继续发展，为人类社会带来更多的便利和挑战。"
        
        print(f"原始文本: {text}")
        result = chain.run(text)
        print(f"摘要: {result}")
    
    def test_translation_chain(self):
        """
        测试翻译链
        """
        print("\n=== 测试翻译链 ===")
        
        chain = self.create_translation_chain()
        
        # 测试文本
        text = "人工智能正在改变我们的生活，它在医疗、金融、教育等领域都有广泛的应用。"
        
        # 翻译成英文
        print(f"原始文本: {text}")
        result_en = chain.run({"text": text, "target_language": "英文"})
        print(f"英文翻译: {result_en}")
        
        # 翻译成日文
        result_ja = chain.run({"text": text, "target_language": "日文"})
        print(f"日文翻译: {result_ja}")
    
    def test_sequential_chain(self):
        """
        测试顺序链
        """
        print("\n=== 测试顺序链 ===")
        
        # 创建链1：内容生成
        chain1 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic"],
                template="请写一篇关于{topic}的短文，不少于300字。"
            ),
            output_key="content"
        )
        
        # 创建链2：摘要生成
        chain2 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content"],
                template="请为以下内容生成一个摘要，不超过100字：\n{content}"
            ),
            output_key="summary"
        )
        
        # 创建链3：翻译
        chain3 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["summary"],
                template="请将以下摘要翻译成英文：\n{summary}"
            ),
            output_key="translation"
        )
        
        # 创建顺序链
        overall_chain = SequentialChain(
            chains=[chain1, chain2, chain3],
            input_variables=["topic"],
            output_variables=["content", "summary", "translation"],
            verbose=True
        )
        
        # 运行链
        result = overall_chain.run("人工智能的伦理问题")
        
        print("\n顺序链结果:")
        print("生成的内容:")
        print(result["content"])
        print("\n生成的摘要:")
        print(result["summary"])
        print("\n英文翻译:")
        print(result["translation"])
    
    def test_chat_chain(self):
        """
        测试聊天链
        """
        print("\n=== 测试聊天链 ===")
        
        # 创建聊天Prompt模板
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的助手，会用中文回答问题。"),
            ("human", "{input}")
        ])
        
        # 创建聊天链
        chain = LLMChain(
            llm=self.llm,
            prompt=chat_prompt,
            memory=self.memory
        )
        
        # 测试多轮对话
        questions = [
            "你好，你是谁？",
            "你能做什么？",
            "你知道什么是人工智能吗？",
            "刚才我问了什么问题？"
        ]
        
        for question in questions:
            print(f"\n用户: {question}")
            result = chain.run(question)
            print(f"助手: {result}")

# 示例用法
def main():
    print("=== LangChain链与模板集成示例 ===")
    
    # 初始化示例
    example = ChainTemplateIntegration()
    
    # 测试问答链
    example.test_qa_chain()
    
    # 测试摘要链
    example.test_summarization_chain()
    
    # 测试翻译链
    example.test_translation_chain()
    
    # 测试顺序链
    example.test_sequential_chain()
    
    # 测试聊天链
    example.test_chat_chain()

if __name__ == "__main__":
    main()