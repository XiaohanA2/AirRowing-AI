import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import Dict, List

# 从analyse_model.py导入必要的类
# 如果analyse_model.py在同一目录下，直接导入
from analyse_model import RowingPostureModel, RowingPostureDataset, AngleCalculator

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
    
    def predict_from_json(self, json_path: str) -> Dict:
        """
        从JSON文件加载数据并预测 - 改进版本
        Args:
            json_path: JSON文件路径
        Returns:
            预测结果列表
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同的JSON格式
            if isinstance(data, dict):
                # 如果是单个对象，转换为列表
                if 'landmarks' in data:
                    data = [data]
                else:
                    # 如果直接是landmarks格式
                    data = [{'landmarks': data}]
            
            for item in data:
                if 'landmarks' in item:
                    landmarks = item['landmarks']
                    
                    # 转换字符串坐标为数字
                    converted_landmarks = {}
                    for key, coords in landmarks.items():
                        if isinstance(coords, list) and len(coords) >= 3:
                            # 如果坐标是字符串，转换为浮点数
                            if isinstance(coords[0], str):
                                converted_landmarks[key] = [float(c) for c in coords[:3]]
                            else:
                                converted_landmarks[key] = coords[:3]
                    
                    result = self.predict_single(converted_landmarks)
            
            return result
            
        except Exception as e:
            print(f"从JSON文件预测时出错: {e}")
            return {}


def evaluate_model_performance(test_data_path: str, model_path: str = 'rowing_model.pth'):
    """
    评估模型性能
    Args:
        test_data_path: 测试数据文件路径
        model_path: 模型文件路径
    """
    model_loader = ModelLoader(model_path)
    result = model_loader.predict_from_json(test_data_path)
    print(f"预测姿态: {result['predicted_posture']}")
    print(f"置信度: {result['confidence']:.3f}")
    print(f"质量评分: {result['quality_score']:.3f}")
    
    print("\n关键指标:")
    print("-" * 30)
    for metric, value in result['angles_and_distances'].items():
        print(f"{metric}: {value:.2f}")
    
    print("\n改进建议:")
    print("-" * 30)
    for advice in result['advice']:
        print(f"• {advice}")
    # if not result:
    #     print("没有有效的测试数据")
    #     return
    
    # # 统计准确率（如果有真实标签）
    # correct_predictions = 0
    # total_predictions = 0
    # quality_scores = []
    
    # for result in results:
    #     if 'true_label' in result:
    #         total_predictions += 1
    #         # 这里需要将姿态名称转换回数字标签进行比较
    #         posture_classes = {
    #             "准备姿态": 0, "抓水阶段": 1, "拉桨阶段": 2,
    #             "出水阶段": 3, "回桨阶段": 4, "错误姿态": 5
    #         }
    #         predicted_label = posture_classes.get(result['predicted_posture'], -1)
    #         if predicted_label == result['true_label']:
    #             correct_predictions += 1
        
    #     quality_scores.append(result['quality_score'])
    
    # print("=" * 50)
    # print("模型性能评估")
    # print("=" * 50)
    
    # if total_predictions > 0:
    #     accuracy = correct_predictions / total_predictions
    #     print(f"准确率: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    # avg_quality = np.mean(quality_scores)
    # print(f"平均质量评分: {avg_quality:.3f}")
    # print(f"质量评分标准差: {np.std(quality_scores):.3f}")
    
    # # 姿态分布统计
    # posture_counts = {}
    # for result in results:
    #     posture = result['predicted_posture']
    #     posture_counts[posture] = posture_counts.get(posture, 0) + 1
    
    # print("\n姿态分布:")
    # print("-" * 30)
    # for posture, count in posture_counts.items():
    #     print(f"{posture}: {count} ({count/len(results)*100:.1f}%)")

# 主函数
if __name__ == "__main__":
    print("赛艇姿态识别模型加载器")
    print("=" * 40)
    
    try:
        evaluate_model_performance('test_data.json')
        
        print("\n" + "=" * 60)
        print("预测完成!")
        
    except FileNotFoundError:
        print("错误: 找不到模型文件 'rowing_model.pth'")
        print("请确保模型文件在正确的路径下")
    except Exception as e:
        print(f"运行时出错: {e}")