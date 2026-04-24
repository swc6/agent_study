# LangChain Prompt模板编写示例

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    FewShotPromptTemplate
)
from langchain_community.llms.dashscope import DashScope
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class PromptTemplatesExample:
    """LangChain Prompt模板编写示例"""
    
    def __init__(self):
        """
        初始化Prompt模板示例
        """
        # 初始化LLM
        self.llm = DashScope(
            model="qwen-plus",
            temperature=0.7
        )
        print("LLM初始化成功！")
    
    def test_basic_prompt(self):
        """
        测试基本Prompt模板
        """
        print("\n=== 测试基本Prompt模板 ===")
        
        # 创建基本Prompt模板
        prompt = PromptTemplate(
            input_variables=["topic", "length"],
            template="请写一篇关于{topic}的短文，不少于{length}字。"
        )
        
        # 格式化Prompt
        formatted_prompt = prompt.format(topic="人工智能", length="200")
        print("格式化后的Prompt:")
        print(formatted_prompt)
        
        # 使用模板生成内容
        response = self.llm.invoke(formatted_prompt)
        print("\n生成的内容:")
        print(response.content)
        
        return response.content
    
    def test_chat_prompt(self):
        """
        测试聊天Prompt模板
        """
        print("\n=== 测试聊天Prompt模板 ===")
        
        # 创建系统消息模板
        system_template = SystemMessagePromptTemplate.from_template(
            "你是一个专业的{field}专家，回答问题要准确专业。"
        )
        
        # 创建人类消息模板
        human_template = HumanMessagePromptTemplate.from_template(
            "请解释{topic}的基本原理。"
        )
        
        # 创建聊天Prompt模板
        chat_prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
        
        # 格式化Prompt
        messages = chat_prompt.format_prompt(
            field="人工智能",
            topic="机器学习"
        ).to_messages()
        
        print("格式化后的消息:")
        for message in messages:
            print(f"{message.type}: {message.content}")
        
        # 使用模板生成内容
        response = self.llm.invoke(messages)
        print("\n生成的内容:")
        print(response.content)
        
        return response.content
    
    def test_few_shot_prompt(self):
        """
        测试少样本Prompt模板
        """
        print("\n=== 测试少样本Prompt模板 ===")
        
        # 示例
        examples = [
            {"input": "如何学习Python", "output": "学习Python的步骤：1. 学习基础语法 2. 实践小项目 3. 学习常用库 4. 参与开源项目"},
            {"input": "如何学习Java", "output": "学习Java的步骤：1. 学习基础语法 2. 理解面向对象 3. 学习Spring框架 4. 实践企业项目"}
        ]
        
        # 示例模板
        example_template = PromptTemplate(
            input_variables=["input", "output"],
            template="问题: {input}\n回答: {output}"
        )
        
        # 创建少样本Prompt模板
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="请按照以下示例的格式回答问题：",
            suffix="问题: {input}\n回答:",
            input_variables=["input"]
        )
        
        # 格式化Prompt
        formatted_prompt = few_shot_prompt.format(input="如何学习人工智能")
        print("格式化后的Prompt:")
        print(formatted_prompt)
        
        # 使用模板生成内容
        response = self.llm.invoke(formatted_prompt)
        print("\n生成的内容:")
        print(response.content)
        
        return response.content
    
    def test_complex_prompt(self):
        """
        测试复杂Prompt模板
        """
        print("\n=== 测试复杂Prompt模板 ===")
        
        # 创建复杂Prompt模板
        prompt = PromptTemplate(
            input_variables=["topic", "requirements", "format"],
            template="""
            请根据以下要求撰写关于{topic}的内容：
            
            要求：
            {requirements}
            
            输出格式：
            {format}
            """
        )
        
        # 格式化参数
        requirements = "\n".join([
            "1. 内容要专业准确",
            "2. 结构清晰，逻辑分明",
            "3. 不少于500字",
            "4. 包含具体示例"
        ])
        
        format = "\n".join([
            "1. 标题：[主题]",
            "2. 简介：[简短介绍]",
            "3. 正文：[详细内容，分点阐述]",
            "4. 结论：[总结性内容]"
        ])
        
        # 格式化Prompt
        formatted_prompt = prompt.format(
            topic="人工智能在金融领域的应用",
            requirements=requirements,
            format=format
        )
        
        print("格式化后的Prompt:")
        print(formatted_prompt)
        
        # 使用模板生成内容
        response = self.llm.invoke(formatted_prompt)
        print("\n生成的内容:")
        print(response.content)
        
        return response.content
    
    def test_dynamic_prompt(self):
        """
        测试动态Prompt模板
        """
        print("\n=== 测试动态Prompt模板 ===")
        
        # 创建动态Prompt模板
        def create_prompt(topic, difficulty):
            """根据难度创建不同的Prompt"""
            if difficulty == "beginner":
                template = "请用简单易懂的语言解释{topic}，适合初学者理解。"
            elif difficulty == "intermediate":
                template = "请详细解释{topic}的原理和应用，适合有一定基础的学习者。"
            else:  # advanced
                template = "请深入分析{topic}的技术细节、最新发展和未来趋势，适合专业人士。"
            
            prompt = PromptTemplate(
                input_variables=["topic"],
                template=template
            )
            
            return prompt.format(topic=topic)
        
        # 测试不同难度的Prompt
        topics = ["机器学习", "深度学习", "强化学习"]
        difficulties = ["beginner", "intermediate", "advanced"]
        
        for topic in topics:
            for difficulty in difficulties:
                print(f"\n{topic} - {difficulty}:")
                prompt = create_prompt(topic, difficulty)
                print(prompt)
                
                # 生成内容（只示例一个）
                if topic == "机器学习" and difficulty == "intermediate":
                    response = self.llm.invoke(prompt)
                    print("\n生成的内容:")
                    print(response.content)
        
        return "动态Prompt测试完成"

# 示例用法
def main():
    print("=== LangChain Prompt模板编写示例 ===")
    
    # 初始化示例
    example = PromptTemplatesExample()
    
    # 测试基本Prompt模板
    example.test_basic_prompt()
    
    # 测试聊天Prompt模板
    example.test_chat_prompt()
    
    # 测试少样本Prompt模板
    example.test_few_shot_prompt()
    
    # 测试复杂Prompt模板
    example.test_complex_prompt()
    
    # 测试动态Prompt模板
    example.test_dynamic_prompt()

if __name__ == "__main__":
    main()