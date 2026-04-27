# FastAPI 接口封装示例

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

# 初始化 FastAPI 应用
app = FastAPI(
    title="Agent API",
    description="Agent 系统的 RESTful API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = None

# 响应模型
class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    thinking: Optional[str] = None
    time_taken: float

# 模拟 Agent
class Agent:
    """模拟 Agent"""
    def process_query(self, query, history=None):
        """处理查询"""
        # 模拟思考过程
        time.sleep(0.5)
        
        # 模拟检索过程
        time.sleep(0.5)
        
        # 模拟生成回答
        time.sleep(0.5)
        
        return {
            "answer": f"这是对 '{query}' 的回答。我是一个智能 Agent，能够回答各种问题。",
            "sources": [
                {"id": "1", "content": "相关内容 1: 人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。"},
                {"id": "2", "content": "相关内容 2: 机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习，而不需要被显式地编程。"}
            ],
            "thinking": "我需要分析用户的查询，然后检索相关信息，最后生成回答。首先，我会理解用户的问题，然后从知识库中检索相关信息，最后根据检索结果生成准确的回答。"
        }

# 创建 Agent 实例
agent = Agent()

# API 路由
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """处理查询请求"""
    try:
        start_time = time.time()
        result = agent.process_query(request.query, request.history)
        time_taken = time.time() - start_time
        result["time_taken"] = time_taken
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {"message": "Agent API 服务运行中"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
