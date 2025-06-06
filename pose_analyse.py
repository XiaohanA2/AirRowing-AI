import json
import numpy as np
import math

def calculate_3d_angle(a, b, c):
    """计算三维空间中三点形成的角度"""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def calculate_torso_lean(shoulder, hip):
    """计算躯干前倾角度"""
    shoulder = np.array(shoulder, dtype=float)
    hip = np.array(hip, dtype=float)
  
    vertical = np.array([shoulder[0], 1, shoulder[2]])
    
    torso_vector = shoulder - hip
    
    cosine_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def analyze_rowing_pose(landmarks):
    """分析划赛艇姿态"""
    
    left_elbow_angle = calculate_3d_angle(
        [float(x) for x in landmarks["LEFT_SHOULDER"]],
        [float(x) for x in landmarks["LEFT_ELBOW"]],
        [float(x) for x in landmarks["LEFT_WRIST"]]
    )
    
    right_elbow_angle = calculate_3d_angle(
        [float(x) for x in landmarks["RIGHT_SHOULDER"]],
        [float(x) for x in landmarks["RIGHT_ELBOW"]],
        [float(x) for x in landmarks["RIGHT_WRIST"]]
    )
    
    left_knee_angle = calculate_3d_angle(
        [float(x) for x in landmarks["LEFT_HIP"]],
        [float(x) for x in landmarks["LEFT_KNEE"]],
        [float(x) for x in landmarks["LEFT_ANKLE"]]
    )
    
    right_knee_angle = calculate_3d_angle(
        [float(x) for x in landmarks["RIGHT_HIP"]],
        [float(x) for x in landmarks["RIGHT_KNEE"]],
        [float(x) for x in landmarks["RIGHT_ANKLE"]]
    )
    
    torso_lean = calculate_torso_lean(
        [float(x) for x in landmarks["LEFT_SHOULDER"]],
        [float(x) for x in landmarks["LEFT_HIP"]]
    )
    
    elbow_symmetry = abs(left_elbow_angle - right_elbow_angle)
    knee_symmetry = abs(left_knee_angle - right_knee_angle)
    
    return {
        "left_elbow_angle": round(left_elbow_angle, 1),
        "right_elbow_angle": round(right_elbow_angle, 1),
        "left_knee_angle": round(left_knee_angle, 1),
        "right_knee_angle": round(right_knee_angle, 1),
        "torso_lean": round(torso_lean, 1),
        "elbow_symmetry": round(elbow_symmetry, 1),
        "knee_symmetry": round(knee_symmetry, 1)
    }

def generate_feedback(analysis):
    feedback = []
    suggestions = []
    
    ideal_elbow = (80, 120)
    if analysis["left_elbow_angle"] < ideal_elbow[0]:
        feedback.append(f"左肘弯曲不足 ({analysis['left_elbow_angle']}° < {ideal_elbow[0]}°)")
        suggestions.append("增加手臂伸展幅度，确保桨叶完全入水")
    elif analysis["left_elbow_angle"] > ideal_elbow[1]:
        feedback.append(f"左肘过度弯曲 ({analysis['left_elbow_angle']}° > {ideal_elbow[1]}°)")
        suggestions.append("减少手臂弯曲，保持前臂与桨杆平行")
    else:
        feedback.append(f"左肘角度理想 ({analysis['left_elbow_angle']}°)")
    if analysis["right_elbow_angle"] < ideal_elbow[0]:
        feedback.append(f"右肘弯曲不足 ({analysis['right_elbow_angle']}° < {ideal_elbow[0]}°)")
        suggestions.append("增加手臂伸展幅度，确保桨叶完全入水")
    elif analysis["right_elbow_angle"] > ideal_elbow[1]:
        feedback.append(f"右肘过度弯曲 ({analysis['right_elbow_angle']}° > {ideal_elbow[1]}°)")
        suggestions.append("减少手臂弯曲，保持前臂与桨杆平行")
    else:
        feedback.append(f"右肘角度理想 ({analysis['right_elbow_angle']}°)")
    
    ideal_knee = (130, 160)
    if analysis["left_knee_angle"] < ideal_knee[0]:
        feedback.append(f"左膝蹬伸不足 ({analysis['left_knee_angle']}° < {ideal_knee[0]}°)")
        suggestions.append("加强腿部发力，确保充分蹬腿")
    elif analysis["left_knee_angle"] > ideal_knee[1]:
        feedback.append(f"左膝过度伸展 ({analysis['left_knee_angle']}° > {ideal_knee[1]}°)")
        suggestions.append("控制蹬腿幅度，避免膝关节锁死")
    else:
        feedback.append(f"左膝角度理想 ({analysis['left_knee_angle']}°)")
    if analysis["right_knee_angle"] < ideal_knee[0]:
        feedback.append(f"右膝蹬伸不足 ({analysis['right_knee_angle']}° < {ideal_knee[0]}°)")
        suggestions.append("加强腿部发力，确保充分蹬腿")
    elif analysis["right_knee_angle"] > ideal_knee[1]:
        feedback.append(f"右膝过度伸展 ({analysis['right_knee_angle']}° > {ideal_knee[1]}°)")
        suggestions.append("控制蹬腿幅度，避免膝关节锁死")
    else:
        feedback.append(f"右膝角度理想 ({analysis['right_knee_angle']}°)")
    
    ideal_torso = (20, 30)
    if analysis["torso_lean"] < ideal_torso[0]:
        feedback.append(f"躯干前倾不足 ({analysis['torso_lean']}° < {ideal_torso[0]}°)")
        suggestions.append("增加身体前倾，更好利用核心力量")
    elif analysis["torso_lean"] > ideal_torso[1]:
        feedback.append(f"躯干过度前倾 ({analysis['torso_lean']}° > {ideal_torso[1]}°)")
        suggestions.append("保持背部挺直，避免腰部过度弯曲")
    else:
        feedback.append(f"躯干前倾角度理想 ({analysis['torso_lean']}°)")
    
    if analysis["elbow_symmetry"] > 15:
        feedback.append(f"左右肘部不对称 ({analysis['elbow_symmetry']}°)")
        suggestions.append("加强弱侧手臂训练，确保双侧力量平衡")
    
    if analysis["knee_symmetry"] > 10:
        feedback.append(f"左右膝盖不对称 ({analysis['knee_symmetry']}°)")
        suggestions.append("检查蹬腿发力均衡性，避免重心偏移")
    
    if not suggestions:
        suggestions.append("动作标准，继续保持！建议增加训练强度")
    
    return feedback, suggestions

def main():
    print("请粘贴MediaPipe输出的JSON坐标数据（输入空行结束）：")
    input_data = []
    while True:
        line = input().strip()
        if line == "":
            break
        input_data.append(line)
    
    json_str = "\n".join(input_data)
    
    try:
        data = json.loads(json_str)
        landmarks = data['landmarks']
        
        # 姿态分析
        analysis = analyze_rowing_pose(landmarks)
        feedback, suggestions = generate_feedback(analysis)
        
        # 输出结果
        print("\n===== 划艇姿态分析报告 =====")
        print(f"左肘角度: {analysis['left_elbow_angle']}°")
        print(f"右肘角度: {analysis['right_elbow_angle']}°")
        print(f"左膝角度: {analysis['left_knee_angle']}°")
        print(f"右膝角度: {analysis['right_knee_angle']}°")
        print(f"躯干前倾: {analysis['torso_lean']}°")
        print(f"肘部对称差: {analysis['elbow_symmetry']}°")
        print(f"膝部对称差: {analysis['knee_symmetry']}°")
        
        print("\n===== 动作评估 =====")
        for item in feedback:
            print(f"• {item}")
        
        print("\n===== 专业建议 =====")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        
        print("\n提示：以上分析基于标准划艇动作生物力学参数")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请确保粘贴完整的JSON格式数据")

if __name__ == "__main__":
    main()
# 输入：
#{
#     "landmarks": {
#         "LEFT_ANKLE": [
#             "0.59215",
#             "0.55969",
#             "-0.14459"
#         ],
#         "LEFT_EAR": [
#             "0.02422",
#             "-0.64221",
#             "-0.08692"
#         ],
#         "LEFT_ELBOW": [
#             "0.20258",
#             "-0.2545",
#             "-0.09864"
#         ],
#         "LEFT_EYE": [
#             "-0.04471",
#             "-0.66508",
#             "-0.16961"
#         ],
#         "LEFT_EYE_INNER": [
#             "-0.04524",
#             "-0.66441",
#             "-0.17024"
#         ],
#         "LEFT_EYE_OUTER": [
#             "-0.04506",
#             "-0.66522",
#             "-0.16938"
#         ],
#         "LEFT_FOOT_INDEX": [
#             "0.60419",
#             "0.65449",
#             "-0.2577"
#         ],
#         "LEFT_HEEL": [
#             "0.63333",
#             "0.59213",
#             "-0.14919"
#         ],
#         "LEFT_HIP": [
#             "0.0935",
#             "-0.01055",
#             "-0.06814"
#         ],
#         "LEFT_INDEX": [
#             "0.34376",
#             "0.04324",
#             "-0.22256"
#         ],
#         "LEFT_KNEE": [
#             "0.24591",
#             "0.32195",
#             "-0.16415"
#         ],
#         "LEFT_PINKY": [
#             "0.36997",
#             "0.03144",
#             "-0.19058"
#         ],
#         "LEFT_SHOULDER": [
#             "0.07682",
#             "-0.47112",
#             "-0.07782"
#         ],
#         "LEFT_THUMB": [
#             "0.32055",
#             "-0.01562",
#             "-0.18669"
#         ],
#         "LEFT_WRIST": [
#             "0.32529",
#             "-0.03613",
#             "-0.17411"
#         ],
#         "MOUTH_LEFT": [
#             "-0.03299",
#             "-0.60508",
#             "-0.15064"
#         ],
#         "MOUTH_RIGHT": [
#             "-0.0737",
#             "-0.60417",
#             "-0.13799"
#         ],
#         "NOSE": [
#             "-0.06102",
#             "-0.63229",
#             "-0.18002"
#         ],
#         "RIGHT_ANKLE": [
#             "-0.14405",
#             "0.57339",
#             "0.22557"
#         ],
#         "RIGHT_EAR": [
#             "-0.11583",
#             "-0.64763",
#             "-0.05353"
#         ],
#         "RIGHT_ELBOW": [
#             "-0.30608",
#             "-0.32025",
#             "0.13767"
#         ],
#         "RIGHT_EYE": [
#             "-0.07549",
#             "-0.66453",
#             "-0.16206"
#         ],
#         "RIGHT_EYE_INNER": [
#             "-0.07541",
#             "-0.66397",
#             "-0.16062"
#         ],
#         "RIGHT_EYE_OUTER": [
#             "-0.07548",
#             "-0.66609",
#             "-0.1605"
#         ],
#         "RIGHT_FOOT_INDEX": [
#             "-0.23753",
#             "0.6831",
#             "0.18636"
#         ],
#         "RIGHT_HEEL": [
#             "-0.13544",
#             "0.61909",
#             "0.23436"
#         ],
#         "RIGHT_HIP": [
#             "-0.09336",
#             "0.01017",
#             "0.06934"
#         ],
#         "RIGHT_INDEX": [
#             "-0.5059",
#             "-0.42707",
#             "0.05309"
#         ],
#         "RIGHT_KNEE": [
#             "-0.26667",
#             "0.23812",
#             "0.07301"
#         ],
#         "RIGHT_PINKY": [
#             "-0.47986",
#             "-0.39773",
#             "0.06523"
#         ],
#         "RIGHT_SHOULDER": [
#             "-0.17228",
#             "-0.46596",
#             "0.07411"
#         ],
#         "RIGHT_THUMB": [
#             "-0.4693",
#             "-0.392",
#             "0.06141"
#         ],
#         "RIGHT_WRIST": [
#             "-0.4589",
#             "-0.38246",
#             "0.06911"
#         ]
#     }
# }

# 输出：
# ===== 划艇姿态分析报告 =====
# 左肘角度: 168.0°
# 右肘角度: 100.9°
# 左膝角度: 144.8°
# 右膝角度: 120.1°
# 躯干前倾: 173.9°
# 肘部对称差: 67.1°
# 膝部对称差: 24.7°

# ===== 动作评估 =====
# • 左肘过度弯曲 (168.0° > 120°)
# • 右肘角度理想 (100.9°)
# • 左膝角度理想 (144.8°)
# • 右膝蹬伸不足 (120.1° < 130°)
# • 躯干过度前倾 (173.9° > 30°)
# • 左右肘部不对称 (67.1°)
# • 左右膝盖不对称 (24.7°)

# ===== 专业建议 =====
# 1. 减少手臂弯曲，保持前臂与桨杆平行
# 2. 加强腿部发力，确保充分蹬腿
# 3. 保持背部挺直，避免腰部过度弯曲
# 4. 加强弱侧手臂训练，确保双侧力量平衡
# 5. 检查蹬腿发力均衡性，避免重心偏移

# 提示：以上分析基于标准划艇动作生物力学参数
