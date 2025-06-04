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
