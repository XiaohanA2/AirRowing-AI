import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import RootModel
from typing import List, Dict
import os
import json


app = FastAPI(title="AirRowing AI Backend")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 从analyse_model.py导入必要的类
# 如果analyse_model.py在同一目录下，直接导入
from analyse_model import RowingPostureModel, RowingPostureDataset, AngleCalculator

class PoseData(RootModel[Dict[str, List[str]]]):
    pass

class ModelLoader:
    """模型加载器"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化模型加载器
        Args:
            model_path: 训练好的模型文件路径 (.pth文件)
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        # 创建模型实例
        self.model = RowingPostureModel()
        
        # 加载训练好的模型权重
        self.load_model()
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"模型已加载到设备: {self.device}")
    
    def load_model(self):
        """加载模型权重"""
        try:
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 加载模型状态字典
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 将模型移动到指定设备
            self.model.to(self.device)
            
            # 打印模型信息
            if 'epoch' in checkpoint:
                print(f"模型训练轮次: {checkpoint['epoch']}")
            if 'val_loss' in checkpoint:
                print(f"验证损失: {checkpoint['val_loss']:.4f}")
            if 'val_accuracy' in checkpoint:
                print(f"验证准确率: {checkpoint['val_accuracy']:.4f}")
                
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
    
    def predict_single(self, landmarks: Dict[str, List[float]]) -> Dict:
        """
        对单个姿态进行预测
        Args:
            landmarks: 关键点坐标字典
        Returns:
            预测结果字典
        """
        return self.model.predict_with_advice(landmarks)
    

def evaluate_model_performance(request: Dict[str, List[float]], model_path: str = 'rowing_model.pth') -> Dict:
    
    model_loader = ModelLoader(model_path)
    result = model_loader.predict_single(request)
    return result
    # print(f"预测姿态: {result['predicted_posture']}")
    # print(f"置信度: {result['confidence']:.3f}")
    # print(f"质量评分: {result['quality_score']:.3f}")
    
    # print("\n关键指标:")
    # print("-" * 30)
    # for metric, value in result['angles_and_distances'].items():
    #     print(f"{metric}: {value:.2f}")
    
    # print("\n改进建议:")
    # print("-" * 30)
    # for advice in result['advice']:
    #     print(f"• {advice}")s



@app.post("/api/analyze-pose")
async def analyze_pose(request: PoseData) -> Dict:
    print("request received:", request)
    converted = {
        k: [float(x) for x in v] for k, v in request.root.items()
    }
    result = evaluate_model_performance(converted)
    
    return result


# 主函数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
    

