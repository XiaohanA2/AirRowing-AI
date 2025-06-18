import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RowingPostureDataset(Dataset):
    """赛艇姿态数据集"""
    
    def __init__(self, data_path: str = None, landmarks_data: List[Dict] = None):
        """
        初始化数据集
        Args:
            data_path: JSON数据文件路径
            landmarks_data: 直接传入的landmarks数据列表
        """
        self.landmark_names = [
            'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER',
            'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX',
            'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST',
            'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR',
            'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER',
            'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX',
            'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'
        ]
        
        # 姿态类别定义
        self.posture_classes = {
            0: "准备姿态",
            1: "抓水阶段", 
            2: "拉桨阶段",
            3: "出水阶段",
            4: "回桨阶段",
            5: "错误姿态"
        }
        
        # 加载数据
        if data_path:
            self.data = self._load_data(data_path)
        elif landmarks_data:
            self.data = landmarks_data
        else:
            # 生成示例数据用于演示
            self.data = self._generate_sample_data()
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """从文件加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_sample_data(self) -> List[Dict]:
        """生成示例训练数据"""
        sample_data = []
        
        # 生成不同姿态的示例数据
        for posture_id in range(6):
            for _ in range(50):  # 每个姿态生成50个样本
                landmarks = {}
                for name in self.landmark_names:
                    # 根据姿态类别生成不同的坐标模式
                    base_noise = np.random.normal(0, 0.1, 3)
                    
                    if posture_id == 0:  # 准备姿态
                        coords = self._get_ready_position_coords(name) + base_noise
                    elif posture_id == 1:  # 抓水阶段
                        coords = self._get_catch_position_coords(name) + base_noise
                    elif posture_id == 2:  # 拉桨阶段
                        coords = self._get_drive_position_coords(name) + base_noise
                    elif posture_id == 3:  # 出水阶段
                        coords = self._get_finish_position_coords(name) + base_noise
                    elif posture_id == 4:  # 回桨阶段
                        coords = self._get_recovery_position_coords(name) + base_noise
                    else:  # 错误姿态
                        coords = np.random.uniform(-1, 1, 3)
                    
                    landmarks[name] = coords.tolist()
                
                sample_data.append({
                    'landmarks': landmarks,
                    'posture_label': posture_id,
                    'quality_score': np.random.uniform(0.3, 1.0) if posture_id < 5 else np.random.uniform(0.0, 0.4)
                })
        
        return sample_data
    
    def _get_ready_position_coords(self, landmark_name: str) -> np.ndarray:
        """获取准备姿态的坐标"""
        # 这里定义理想的准备姿态各关键点位置
        ready_positions = {
            'LEFT_SHOULDER': np.array([0.1, -0.4, -0.08]),
            'RIGHT_SHOULDER': np.array([-0.1, -0.4, 0.08]),
            'LEFT_ELBOW': np.array([0.2, -0.2, -0.1]),
            'RIGHT_ELBOW': np.array([-0.2, -0.2, 0.1]),
            'LEFT_WRIST': np.array([0.3, 0.0, -0.15]),
            'RIGHT_WRIST': np.array([-0.3, 0.0, 0.15]),
            'LEFT_HIP': np.array([0.1, 0.0, -0.05]),
            'RIGHT_HIP': np.array([-0.1, 0.0, 0.05]),
            'LEFT_KNEE': np.array([0.2, 0.3, -0.1]),
            'RIGHT_KNEE': np.array([-0.2, 0.3, 0.1]),
        }
        return ready_positions.get(landmark_name, np.random.uniform(-0.5, 0.5, 3))
    
    def _get_catch_position_coords(self, landmark_name: str) -> np.ndarray:
        """获取抓水阶段的坐标"""
        catch_positions = {
            'LEFT_SHOULDER': np.array([0.08, -0.45, -0.08]),
            'RIGHT_SHOULDER': np.array([-0.08, -0.45, 0.08]),
            'LEFT_ELBOW': np.array([0.15, -0.3, -0.12]),
            'RIGHT_ELBOW': np.array([-0.15, -0.3, 0.12]),
            'LEFT_WRIST': np.array([0.25, -0.1, -0.18]),
            'RIGHT_WRIST': np.array([-0.25, -0.1, 0.18]),
        }
        return catch_positions.get(landmark_name, np.random.uniform(-0.5, 0.5, 3))
    
    def _get_drive_position_coords(self, landmark_name: str) -> np.ndarray:
        """获取拉桨阶段的坐标"""
        drive_positions = {
            'LEFT_SHOULDER': np.array([0.12, -0.42, -0.06]),
            'RIGHT_SHOULDER': np.array([-0.12, -0.42, 0.06]),
            'LEFT_ELBOW': np.array([0.25, -0.15, -0.08]),
            'RIGHT_ELBOW': np.array([-0.25, -0.15, 0.08]),
            'LEFT_WRIST': np.array([0.35, 0.1, -0.12]),
            'RIGHT_WRIST': np.array([-0.35, 0.1, 0.12]),
        }
        return drive_positions.get(landmark_name, np.random.uniform(-0.5, 0.5, 3))
    
    def _get_finish_position_coords(self, landmark_name: str) -> np.ndarray:
        """获取出水阶段的坐标"""
        finish_positions = {
            'LEFT_SHOULDER': np.array([0.15, -0.38, -0.04]),
            'RIGHT_SHOULDER': np.array([-0.15, -0.38, 0.04]),
            'LEFT_ELBOW': np.array([0.3, -0.05, -0.05]),
            'RIGHT_ELBOW': np.array([-0.3, -0.05, 0.05]),
            'LEFT_WRIST': np.array([0.4, 0.2, -0.08]),
            'RIGHT_WRIST': np.array([-0.4, 0.2, 0.08]),
        }
        return finish_positions.get(landmark_name, np.random.uniform(-0.5, 0.5, 3))
    
    def _get_recovery_position_coords(self, landmark_name: str) -> np.ndarray:
        """获取回桨阶段的坐标"""
        recovery_positions = {
            'LEFT_SHOULDER': np.array([0.08, -0.41, -0.07]),
            'RIGHT_SHOULDER': np.array([-0.08, -0.41, 0.07]),
            'LEFT_ELBOW': np.array([0.18, -0.25, -0.09]),
            'RIGHT_ELBOW': np.array([-0.18, -0.25, 0.09]),
            'LEFT_WRIST': np.array([0.28, -0.05, -0.14]),
            'RIGHT_WRIST': np.array([-0.28, -0.05, 0.14]),
        }
        return recovery_positions.get(landmark_name, np.random.uniform(-0.5, 0.5, 3))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        landmarks = item['landmarks']
        
        # 转换为固定长度的向量
        feature_vector = []
        for name in self.landmark_names:
            if name in landmarks:
                coords = landmarks[name]
                if isinstance(coords, list) and len(coords) >= 3:
                    feature_vector.extend(coords[:3])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
        
        features = torch.tensor(feature_vector, dtype=torch.float32)
        
        # 标签处理
        posture_label = item.get('posture_label', 0)
        quality_score = item.get('quality_score', 0.5)
        
        return {
            'features': features,
            'posture_label': torch.tensor(posture_label, dtype=torch.long),
            'quality_score': torch.tensor(quality_score, dtype=torch.float32)
        }


class RowingPostureModel(nn.Module):
    """赛艇姿态识别模型"""
    
    def __init__(self, input_dim: int = 99, hidden_dims: List[int] = [512, 256, 128], 
                 num_posture_classes: int = 6, dropout_rate: float = 0.3):
        super(RowingPostureModel, self).__init__()
        
        self.input_dim = input_dim  # 33个关键点 × 3个坐标 = 99
        self.num_posture_classes = num_posture_classes
        
        # 特征提取层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 姿态分类头
        self.posture_classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_posture_classes)
        )
        
        # 质量评分头
        self.quality_regressor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间的质量分数
        )
        
        # 角度特征提取（用于计算关键角度）
        self.angle_calculator = AngleCalculator()
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 姿态分类
        posture_logits = self.posture_classifier(features)
        
        # 质量评分
        quality_score = self.quality_regressor(features)
        
        return {
            'posture_logits': posture_logits,
            'quality_score': quality_score.squeeze(-1),
            'features': features
        }
    
    def predict_with_advice(self, landmarks: Dict[str, List[float]]) -> Dict:
        """预测姿态并提供建议"""
        self.eval()
        
        # 准备输入数据
        dataset = RowingPostureDataset(landmarks_data=[{'landmarks': landmarks}])
        sample = dataset[0]
        features = sample['features'].unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.forward(features)
            
            # 获取预测结果
            posture_probs = F.softmax(outputs['posture_logits'], dim=1)
            predicted_posture = torch.argmax(posture_probs, dim=1).item()
            confidence = posture_probs[0][predicted_posture].item()
            quality_score = outputs['quality_score'].item()
            
            # 计算关键角度和距离
            angles_and_distances = self.angle_calculator.calculate_metrics(landmarks)
            
            # 生成建议
            advice = self._generate_advice(predicted_posture, quality_score, angles_and_distances)
            
            return {
                'predicted_posture': dataset.posture_classes[predicted_posture],
                'confidence': confidence,
                'quality_score': quality_score,
                'angles_and_distances': angles_and_distances,
                'advice': advice
            }
    
    def _generate_advice(self, posture_id: int, quality_score: float, metrics: Dict) -> List[str]:
        """根据预测结果生成建议"""
        advice = []
        
        # 基于姿态类别的建议
        posture_advice = {
            0: ["保持身体直立", "双肩放松下沉", "膝盖微屈"],
            1: ["前倾幅度适中", "手臂充分伸展", "保持核心稳定"],
            2: ["腿部主导发力", "保持背部挺直", "手臂跟随腿部动作"],
            3: ["手柄拉至胸部", "肘部贴近身体", "保持肩膀稳定"],
            4: ["缓慢回桨", "手臂先伸展", "身体后续跟随"],
            5: ["检查整体姿态", "注意动作协调性", "建议回到基本动作练习"]
        }
        
        advice.extend(posture_advice.get(posture_id, ["保持标准姿态"]))
        
        # 基于质量分数的建议
        if quality_score < 0.5:
            advice.append("整体动作质量需要改善，建议加强基础训练")
        elif quality_score < 0.7:
            advice.append("动作基本正确，注意细节优化")
        else:
            advice.append("动作质量良好，继续保持")
        
        # 基于角度和距离的建议
        if 'knee_angle' in metrics:
            knee_angle = metrics['knee_angle']
            if knee_angle < 90:
                advice.append("膝盖弯曲过度，适当减少弯曲角度")
            elif knee_angle > 160:
                advice.append("膝盖伸展不足，增加弯曲幅度")
        
        if 'back_angle' in metrics:
            back_angle = metrics['back_angle']
            if back_angle < 80:
                advice.append("身体前倾过度，适当挺直背部")
            elif back_angle > 110:
                advice.append("身体过于直立，适当增加前倾")
        
        return advice


class AngleCalculator:
    """角度和距离计算器"""
    
    def calculate_angle(self, p1: List[float], p2: List[float], p3: List[float]) -> float:
        """计算三点构成的角度"""
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def calculate_distance(self, p1: List[float], p2: List[float]) -> float:
        """计算两点间距离"""
        p1, p2 = np.array(p1), np.array(p2)
        return np.linalg.norm(p1 - p2)
    
    def calculate_metrics(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """计算关键的角度和距离指标"""
        metrics = {}
        
        try:
            # 膝盖角度（大腿-小腿）
            if all(k in landmarks for k in ['LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE']):
                metrics['left_knee_angle'] = self.calculate_angle(
                    landmarks['LEFT_HIP'], landmarks['LEFT_KNEE'], landmarks['LEFT_ANKLE']
                )
            
            if all(k in landmarks for k in ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']):
                metrics['right_knee_angle'] = self.calculate_angle(
                    landmarks['RIGHT_HIP'], landmarks['RIGHT_KNEE'], landmarks['RIGHT_ANKLE']
                )
            
            # 肘部角度
            if all(k in landmarks for k in ['LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST']):
                metrics['left_elbow_angle'] = self.calculate_angle(
                    landmarks['LEFT_SHOULDER'], landmarks['LEFT_ELBOW'], landmarks['LEFT_WRIST']
                )
            
            if all(k in landmarks for k in ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST']):
                metrics['right_elbow_angle'] = self.calculate_angle(
                    landmarks['RIGHT_SHOULDER'], landmarks['RIGHT_ELBOW'], landmarks['RIGHT_WRIST']
                )
            
            # 身体前倾角度（简化计算）
            if all(k in landmarks for k in ['LEFT_SHOULDER', 'LEFT_HIP']):
                shoulder_hip_vector = np.array(landmarks['LEFT_SHOULDER']) - np.array(landmarks['LEFT_HIP'])
                vertical_vector = np.array([0, -1, 0])
                
                cos_angle = np.dot(shoulder_hip_vector, vertical_vector) / (
                    np.linalg.norm(shoulder_hip_vector) * np.linalg.norm(vertical_vector)
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                metrics['back_angle'] = np.degrees(np.arccos(cos_angle))
            
            # 双手距离
            if all(k in landmarks for k in ['LEFT_WRIST', 'RIGHT_WRIST']):
                metrics['hand_distance'] = self.calculate_distance(
                    landmarks['LEFT_WRIST'], landmarks['RIGHT_WRIST']
                )
            
            # 膝盖平均角度
            if 'left_knee_angle' in metrics and 'right_knee_angle' in metrics:
                metrics['knee_angle'] = (metrics['left_knee_angle'] + metrics['right_knee_angle']) / 2
            
        except Exception as e:
            print(f"计算指标时出错: {e}")
        
        return metrics


class RowingTrainer:
    """赛艇模型训练器"""
    
    def __init__(self, model: RowingPostureModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 损失函数
        self.posture_criterion = nn.CrossEntropyLoss()
        self.quality_criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        posture_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            posture_labels = batch['posture_label'].to(self.device)
            quality_scores = batch['quality_score'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(features)
            
            # 计算损失
            posture_loss = self.posture_criterion(outputs['posture_logits'], posture_labels)
            quality_loss = self.quality_criterion(outputs['quality_score'], quality_scores)
            
            # 总损失（可以调整权重）
            total_batch_loss = posture_loss + 0.5 * quality_loss
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs['posture_logits'], 1)
            posture_correct += (predicted == posture_labels).sum().item()
            total_samples += posture_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = posture_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        posture_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                posture_labels = batch['posture_label'].to(self.device)
                quality_scores = batch['quality_score'].to(self.device)
                
                outputs = self.model(features)
                
                posture_loss = self.posture_criterion(outputs['posture_logits'], posture_labels)
                quality_loss = self.quality_criterion(outputs['quality_score'], quality_scores)
                total_batch_loss = posture_loss + 0.5 * quality_loss
                
                total_loss += total_batch_loss.item()
                
                _, predicted = torch.max(outputs['posture_logits'], 1)
                posture_correct += (predicted == posture_labels).sum().item()
                total_samples += posture_labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = posture_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
              num_epochs: int = 100, save_path: str = 'rowing_model.pth'):
        """训练模型"""
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print("开始训练...")
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_dataloader)
            
            # 验证
            val_metrics = self.validate(val_dataloader)
            
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_metrics['loss'])
            
            # 学习率调度
            self.scheduler.step(val_metrics['loss'])
            
            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                }, save_path)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print("-" * 50)
        
        print("训练完成!")
        return train_losses, val_losses


# 使用示例
def main():
    """主函数 - 演示如何使用模型"""
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建数据集...")
    full_dataset = RowingPostureDataset()
    
    # 数据集分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = RowingPostureModel()
    
    # 创建训练器
    trainer = RowingTrainer(model, device)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, num_epochs=50)
    
    # 测试预测功能
    print("\n测试预测功能...")
    
    # 使用提供的示例数据
    sample_landmarks = {
        "LEFT_ANKLE": [0.59215, 0.55969, -0.14459],
        "LEFT_EAR": [0.02422, -0.64221, -0.08692],
        "LEFT_ELBOW": [0.20258, -0.2545, -0.09864],
        "LEFT_EYE": [-0.04471, -0.66508, -0.16961],
        "LEFT_EYE_INNER": [-0.04524, -0.66441, -0.17024],
        "LEFT_EYE_OUTER": [-0.04506, -0.66522, -0.16938],
        "LEFT_FOOT_INDEX": [0.60419, 0.65449, -0.2577],
        "LEFT_HEEL": [0.63333, 0.59213, -0.14919],
        "LEFT_HIP": [0.0935, -0.01055, -0.06814],
        "LEFT_INDEX": [0.34376, 0.04324, -0.22256],
        "LEFT_KNEE": [0.24591, 0.32195, -0.16415],
        "LEFT_PINKY": [0.36997, 0.03144, -0.19058],
        "LEFT_SHOULDER": [0.07682, -0.47112, -0.07782],
        "LEFT_THUMB": [0.32055, -0.01562, -0.18669],
        "LEFT_WRIST": [0.32529, -0.03613, -0.17411],
        "MOUTH_LEFT": [-0.03299, -0.60508, -0.15064],
        "MOUTH_RIGHT": [-0.0737, -0.60417, -0.13799],
        "NOSE": [-0.06102, -0.63229, -0.18002],
        "RIGHT_ANKLE": [-0.14405, 0.57339, 0.22557],
        "RIGHT_EAR": [-0.11583, -0.64763, -0.05353],
        "RIGHT_ELBOW": [-0.30608, -0.32025, 0.13767],
        "RIGHT_EYE": [-0.07549, -0.66453, -0.16206],
        "RIGHT_EYE_INNER": [-0.07541, -0.66397, -0.16062],
        "RIGHT_EYE_OUTER": [-0.07548, -0.66609, -0.1605],
        "RIGHT_FOOT_INDEX": [-0.23753, 0.6831, 0.18636],
        "RIGHT_HEEL": [-0.13544, 0.61909, 0.23436],
        "RIGHT_HIP": [-0.09336, 0.01017, 0.06934],
        "RIGHT_INDEX": [-0.5059, -0.42707, 0.05309],
        "RIGHT_KNEE": [-0.26667, 0.23812, 0.07301],
        "RIGHT_PINKY": [-0.47986, -0.39773, 0.06523],
        "RIGHT_SHOULDER": [-0.17228, -0.46596, 0.07411],
        "RIGHT_THUMB": [-0.4693, -0.392, 0.06141],
        "RIGHT_WRIST": [-0.4589, -0.38246, 0.06911]
    }
    
    # 进行预测
    prediction_result = model.predict_with_advice(sample_landmarks)
    
    print("预测结果:")
    print(f"预测姿态: {prediction_result['predicted_posture']}")
    print(f"置信度: {prediction_result['confidence']:.3f}")
    print(f"质量评分: {prediction_result['quality_score']:.3f}")
    
    print("\n关键指标:")
    for metric, value in prediction_result['angles_and_distances'].items():
        print(f"{metric}: {value:.2f}")
    
    print("\n改进建议:")
    for advice in prediction_result['advice']:
        print(f"- {advice}")
    
    print("\n模型保存完成!")


def load_and_predict(model_path: str, landmarks_data: Dict[str, List[float]]):
    """加载训练好的模型并进行预测"""
    
    # 创建模型
    model = RowingPostureModel()
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 进行预测
    result = model.predict_with_advice(landmarks_data)
    
    return result


def create_custom_dataset(landmarks_list: List[Dict], labels: List[int], quality_scores: List[float]):
    """创建自定义数据集"""
    
    data = []
    for landmarks, label, quality in zip(landmarks_list, labels, quality_scores):
        data.append({
            'landmarks': landmarks,
            'posture_label': label,
            'quality_score': quality
        })
    
    return RowingPostureDataset(landmarks_data=data)


def analyze_rowing_technique(landmarks_sequence: List[Dict[str, List[float]]]):
    """分析一组连续的划船动作"""
    
    model = RowingPostureModel()
    # 这里应该加载预训练的模型
    # model.load_state_dict(torch.load('rowing_model.pth')['model_state_dict'])
    
    results = []
    for i, landmarks in enumerate(landmarks_sequence):
        result = model.predict_with_advice(landmarks)
        result['frame'] = i
        results.append(result)
    
    # 分析整体表现
    quality_scores = [r['quality_score'] for r in results]
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    # 统计各姿态出现次数
    posture_counts = {}
    for result in results:
        posture = result['predicted_posture']
        posture_counts[posture] = posture_counts.get(posture, 0) + 1
    
    analysis = {
        'individual_results': results,
        'average_quality': avg_quality,
        'posture_distribution': posture_counts,
        'improvement_suggestions': generate_sequence_advice(results)
    }
    
    return analysis


def generate_sequence_advice(results: List[Dict]) -> List[str]:
    """基于动作序列生成改进建议"""
    
    advice = []
    
    # 检查质量分数变化
    quality_scores = [r['quality_score'] for r in results]
    if len(quality_scores) > 1:
        quality_trend = quality_scores[-1] - quality_scores[0]
        if quality_trend > 0.1:
            advice.append("动作质量在提升，继续保持")
        elif quality_trend < -0.1:
            advice.append("动作质量在下降，注意疲劳管理")
    
    # 检查姿态一致性
    postures = [r['predicted_posture'] for r in results]
    if len(set(postures)) == 1:
        advice.append("姿态保持一致，很好")
    else:
        advice.append("姿态变化较大，注意动作稳定性")
    
    # 检查错误姿态
    error_count = postures.count("错误姿态")
    if error_count > len(postures) * 0.2:
        advice.append("错误姿态较多，建议回到基础训练")
    
    return advice


# 模型配置类
class ModelConfig:
    """模型配置类"""
    
    def __init__(self):
        self.input_dim = 99  # 33个关键点 × 3个坐标
        self.hidden_dims = [512, 256, 128]
        self.num_posture_classes = 6
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 姿态类别定义
        self.posture_classes = {
            0: "准备姿态",
            1: "抓水阶段", 
            2: "拉桨阶段",
            3: "出水阶段",
            4: "回桨阶段",
            5: "错误姿态"
        }
        
        # 关键点名称
        self.landmark_names = [
            'LEFT_ANKLE', 'LEFT_EAR', 'LEFT_ELBOW', 'LEFT_EYE', 'LEFT_EYE_INNER',
            'LEFT_EYE_OUTER', 'LEFT_FOOT_INDEX', 'LEFT_HEEL', 'LEFT_HIP', 'LEFT_INDEX',
            'LEFT_KNEE', 'LEFT_PINKY', 'LEFT_SHOULDER', 'LEFT_THUMB', 'LEFT_WRIST',
            'MOUTH_LEFT', 'MOUTH_RIGHT', 'NOSE', 'RIGHT_ANKLE', 'RIGHT_EAR',
            'RIGHT_ELBOW', 'RIGHT_EYE', 'RIGHT_EYE_INNER', 'RIGHT_EYE_OUTER',
            'RIGHT_FOOT_INDEX', 'RIGHT_HEEL', 'RIGHT_HIP', 'RIGHT_INDEX',
            'RIGHT_KNEE', 'RIGHT_PINKY', 'RIGHT_SHOULDER', 'RIGHT_THUMB', 'RIGHT_WRIST'
        ]


class RowingPostureEvaluator:
    """赛艇姿态评估器"""
    
    def __init__(self, model: RowingPostureModel):
        self.model = model
        self.angle_calculator = AngleCalculator()
    
    def evaluate_posture_quality(self, landmarks: Dict[str, List[float]]) -> Dict:
        """详细评估姿态质量"""
        
        # 计算关键指标
        metrics = self.angle_calculator.calculate_metrics(landmarks)
        
        # 评估各个方面
        evaluation = {
            'body_alignment': self._evaluate_body_alignment(landmarks, metrics),
            'leg_position': self._evaluate_leg_position(landmarks, metrics),
            'arm_position': self._evaluate_arm_position(landmarks, metrics),
            'overall_balance': self._evaluate_overall_balance(landmarks, metrics)
        }
        
        # 计算总体评分
        scores = [eval_item['score'] for eval_item in evaluation.values()]
        overall_score = sum(scores) / len(scores)
        
        evaluation['overall_score'] = overall_score
        evaluation['detailed_feedback'] = self._generate_detailed_feedback(evaluation)
        
        return evaluation
    
    def _evaluate_body_alignment(self, landmarks: Dict, metrics: Dict) -> Dict:
        """评估身体对齐"""
        score = 1.0
        issues = []
        
        # 检查背部角度
        if 'back_angle' in metrics:
            back_angle = metrics['back_angle']
            if back_angle < 70 or back_angle > 120:
                score -= 0.3
                issues.append("背部角度不理想")
        
        # 检查肩膀水平
        if all(k in landmarks for k in ['LEFT_SHOULDER', 'RIGHT_SHOULDER']):
            left_shoulder_y = landmarks['LEFT_SHOULDER'][1]
            right_shoulder_y = landmarks['RIGHT_SHOULDER'][1]
            shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
            
            if shoulder_diff > 0.05:
                score -= 0.2
                issues.append("肩膀不平衡")
        
        return {'score': max(0, score), 'issues': issues}
    
    def _evaluate_leg_position(self, landmarks: Dict, metrics: Dict) -> Dict:
        """评估腿部位置"""
        score = 1.0
        issues = []
        
        # 检查膝盖角度
        if 'knee_angle' in metrics:
            knee_angle = metrics['knee_angle']
            if knee_angle < 60 or knee_angle > 170:
                score -= 0.4
                issues.append("膝盖角度不合适")
        
        # 检查脚踝位置
        if all(k in landmarks for k in ['LEFT_ANKLE', 'RIGHT_ANKLE']):
            left_ankle = landmarks['LEFT_ANKLE']
            right_ankle = landmarks['RIGHT_ANKLE']
            ankle_distance = abs(left_ankle[0] - right_ankle[0])
            
            if ankle_distance > 0.3:
                score -= 0.2
                issues.append("脚踝位置不对称")
        
        return {'score': max(0, score), 'issues': issues}
    
    def _evaluate_arm_position(self, landmarks: Dict, metrics: Dict) -> Dict:
        """评估手臂位置"""
        score = 1.0
        issues = []
        
        # 检查肘部角度
        left_elbow_angle = metrics.get('left_elbow_angle', 90)
        right_elbow_angle = metrics.get('right_elbow_angle', 90)
        
        if abs(left_elbow_angle - right_elbow_angle) > 20:
            score -= 0.3
            issues.append("左右手臂角度不一致")
        
        # 检查手腕位置
        if 'hand_distance' in metrics:
            hand_distance = metrics['hand_distance']
            if hand_distance < 0.1 or hand_distance > 0.6:
                score -= 0.2
                issues.append("双手距离不合适")
        
        return {'score': max(0, score), 'issues': issues}
    
    def _evaluate_overall_balance(self, landmarks: Dict, metrics: Dict) -> Dict:
        """评估整体平衡"""
        score = 1.0
        issues = []
        
        # 检查重心位置（简化）
        if all(k in landmarks for k in ['LEFT_HIP', 'RIGHT_HIP']):
            left_hip = landmarks['LEFT_HIP']
            right_hip = landmarks['RIGHT_HIP']
            
            # 检查髋部水平
            hip_height_diff = abs(left_hip[1] - right_hip[1])
            if hip_height_diff > 0.05:
                score -= 0.2
                issues.append("髋部不平衡")
            
            # 检查左右对称性
            hip_lateral_diff = abs(left_hip[0] - (-right_hip[0]))
            if hip_lateral_diff > 0.05:
                score -= 0.1
                issues.append("左右不对称")
        
        return {'score': max(0, score), 'issues': issues}
    
    def _generate_detailed_feedback(self, evaluation: Dict) -> List[str]:
        """生成详细反馈"""
        feedback = []
        
        for aspect, eval_data in evaluation.items():
            if aspect == 'overall_score' or aspect == 'detailed_feedback':
                continue
                
            if eval_data['score'] < 0.7:
                feedback.append(f"{aspect}需要改进:")
                feedback.extend([f"  - {issue}" for issue in eval_data['issues']])
            elif eval_data['score'] < 0.9:
                feedback.append(f"{aspect}基本良好，注意细节优化")
        
        if not feedback:
            feedback.append("整体姿态良好，继续保持!")
        
        return feedback


if __name__ == "__main__":
    main()