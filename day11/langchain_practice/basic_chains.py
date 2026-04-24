# LangChain 基础链使用示例

import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_community.llms.dashscope import DashScope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class BasicChainsExample:
    """LangChain基础链使用示例"""
    
    def __init__(self):
        """
        初始化LangChain基础链示例
        """
        # 初始化LLM
        self.llm = DashScope(
            model="qwen-plus",
            temperature=0.7,
            max_tokens=1000
        )
        print("LLM初始化成功！")
    
    def test_llm_chain(self):
        """
        测试LLMChain
        """
        print("\n=== 测试LLMChain ===")
        
        # 创建提示词模板
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="请写一篇关于{topic}的短文，不少于200字。"
        )
        
        # 创建LLMChain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        # 运行链
        result = chain.run("人工智能的未来")
        print("LLMChain结果:")
        print(result)
        
        return result
    
    def test_simple_sequential_chain(self):
        """
        测试SimpleSequentialChain
        """
        print("\n=== 测试SimpleSequentialChain ===")
        
        # 创建第一个链：生成标题
        prompt1 = PromptTemplate(
            input_variables=["topic"],
            template="请为关于{topic}的文章生成一个吸引人的标题。"
        )
        chain1 = LLMChain(llm=self.llm, prompt=prompt1)
        
        # 创建第二个链：根据标题生成内容
        prompt2 = PromptTemplate(
            input_variables=["title"],
            template="请根据标题 '{title}' 写一篇短文，不少于200字。"
        )
        chain2 = LLMChain(llm=self.llm, prompt=prompt2)
        
        # 创建简单顺序链
        overall_chain = SimpleSequentialChain(
            chains=[chain1, chain2],
            verbose=True
        )
        
        # 运行链
        result = overall_chain.run("人工智能在医疗领域的应用")
        print("SimpleSequentialChain结果:")
        print(result)
        
        return result
    
    def test_sequential_chain(self):
        """
        测试SequentialChain
        """
        print("\n=== 测试SequentialChain ===")
        
        # 创建链1：生成标题
        chain1 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic"],
                template="请为关于{topic}的文章生成5个吸引人的标题。"
            ),
            output_key="titles"
        )
        
        # 创建链2：选择最佳标题
        chain2 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["titles"],
                template="请从以下标题中选择最佳的一个，并说明理由：\n{titles}"
            ),
            output_key="best_title"
        )
        
        # 创建链3：生成内容
        chain3 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["best_title", "topic"],
                template="请根据标题 '{best_title}' 写一篇关于{topic}的文章，不少于300字。"
            ),
            output_key="content"
        )
        
        # 创建链4：生成摘要
        chain4 = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content"],
                template="请为以下文章生成一个100字以内的摘要：\n{content}"
            ),
            output_key="summary"
        )
        
        # 创建顺序链
        overall_chain = SequentialChain(
            chains=[chain1, chain2, chain3, chain4],
            input_variables=["topic"],
            output_variables=["titles", "best_title", "content", "summary"],
            verbose=True
        )
        
        # 运行链
        result = overall_chain.run("人工智能在教育领域的应用")
        
        print("\nSequentialChain结果:")
        print("标题列表:")
        print(result["titles"])
        print("\n最佳标题:")
        print(result["best_title"])
        print("\n文章内容:")
        print(result["content"])
        print("\n摘要:")
        print(result["summary"])
        
        return result
    
    def test_batch_processing(self):
        """
        测试批量处理
        """
        print("\n=== 测试批量处理 ===")
        
        # 创建提示词模板
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="请写一个关于{topic}的简短介绍，不超过100字。"
        )
        
        # 创建链
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # 批量输入
        inputs = [
            {"topic": "人工智能"},
            {"topic": "机器学习"},
            {"topic": "深度学习"}
        ]
        
        # 执行批量处理
        results = chain.batch(inputs)
        
        print("批量处理结果:")
        for i, result in enumerate(results):
            print(f"\n{inputs[i]['topic']}:")
            print(result["text"])
        
        return results

# 示例用法
def main():
    print("=== LangChain基础链使用示例 ===")
    
    # 初始化示例
    example = BasicChainsExample()
    
    # 测试LLMChain
    example.test_llm_chain()
    
    # 测试SimpleSequentialChain
    example.test_simple_sequential_chain()
    
    # 测试SequentialChain
    example.test_sequential_chain()
    
    # 测试批量处理
    example.test_batch_processing()

if __name__ == "__main__":
    main()