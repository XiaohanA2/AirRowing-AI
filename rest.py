from flask import Flask, request, jsonify
import cv2
import tempfile
import os
from mediapipe.python.solutions import pose as mp_pose


class PoseTracker:
    landmark_names = [name for name in mp_pose.PoseLandmark.__members__.keys()]

    def __init__(self):
        self.pose = None

    def __enter__(self):
        # 初始化模型
        self.pose = mp_pose.Pose()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pose:
            self.pose.close()

    def get_landmarks(self, image):
        input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 运行模型获取结果
        result = self.pose.process(image=input_frame)
        pose_landmarks = result.pose_world_landmarks
    
        if pose_landmarks is None:
            return None

        assert len(pose_landmarks.landmark) == 33, f'Unexpected number of predicted pose landmarks: {len(pose_landmarks.landmark)}'

        # 格式化标记点：特征点名-特征点坐标
        pose_landmarks = {
            name: [str(round(lmk.x, 5)), str(round(lmk.y, 5)), str(round(lmk.z, 5))]
            for name, lmk in zip(self.landmark_names, pose_landmarks.landmark)
        }

        return pose_landmarks


app = Flask(__name__)

@app.route('/pose', methods=['POST'])
def detect_pose():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 401

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty image file"}), 401

    try:
        # 保存图片到临时文件中，便于后续处理
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, 'input.jpg')
        file.save(temp_path)

        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({"error": "Failed to read image"}), 400

        # 调用模型获取特征点坐标
        landmarks = tracker.get_landmarks(image)
        if landmarks is None:
            return jsonify({"error": "No human detected in the image."}), 400

        return jsonify({
            "landmarks": landmarks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    with PoseTracker() as tracker:
        app.run(host='0.0.0.0', port=5000, debug=True)

# 前端调用示例：
# <template>
#   <input type="file" accept="image/*" @change="handleImageUpload" />
# </template>
# 
# <script setup>
# import { ref } from 'vue'
# 
# const landmarks = ref(null)
# const error = ref(null)
# 
# const handleImageUpload = async (event) => {
#   const file = event.target.files[0]
#   if (!file) return
# 
#   const formData = new FormData()
#   formData.append('image', file)
# 
#   landmarks.value = null
#   error.value = null
# 
#   try {
#     const response = await fetch('http://localhost:5000/pose', {
#       method: 'POST',
#       body: formData
#     })
# 
#     const result = await response.json()
# 
#     if (!response.ok) {
#       throw new Error(result.error || '上传失败')
#     }
# 
#     landmarks.value = result.landmarks
#   } catch (err) {
#     error.value = err.message
#   }
# }
# </script>

# 返回结果示例：
# {
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
