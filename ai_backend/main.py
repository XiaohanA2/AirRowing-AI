from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

# 加载环境变量
load_dotenv()

app = FastAPI(title="AirRowing AI Backend")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置DeepSeek API
client = AsyncOpenAI(
    api_key="sk-4d4e5befec2c4210951c385aea1611af",
    base_url="https://api.deepseek.com"
)

class PoseData(BaseModel):
    keypoints: List[dict]
    model: str
    user_id: str

class AnalysisRequest(BaseModel):
    pose_data: PoseData
    question: Optional[str] = None

@app.post("/api/analyze-pose")
async def analyze_pose(request: AnalysisRequest):
    try:
        prompt = build_prompt(request.pose_data)
        response = await call_deepseek_api(prompt)
        return {
            "success": True,
            "analysis": response,
            "model": request.pose_data.model
        }
    except Exception as e:
        import traceback
        print("后端异常：", e)
        traceback.print_exc()  # 打印详细堆栈
        raise HTTPException(status_code=500, detail=str(e))

def build_prompt(pose_data: PoseData) -> str:
    """构建发送给AI的提示词"""
    prompt = f"""作为一名专业的赛艇教练，请分析以下赛艇训练姿势数据并提供专业建议：

姿态数据：
{json.dumps(pose_data.keypoints, indent=2)}

请从以下几个方面进行分析：
1. 整体姿态评估
2. 具体动作要点分析
3. 存在的问题
4. 改进建议
5. 训练建议

请用专业但易懂的语言回答。"""
    
    return prompt

async def call_deepseek_api(prompt: str) -> str:
    print('调用DeepSeek，prompt内容：', prompt)
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的赛艇教练，擅长分析训练姿势并提供改进建议。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        import traceback
        print('DeepSeek API调用异常：', e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DeepSeek API调用失败: {str(e)}")

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 