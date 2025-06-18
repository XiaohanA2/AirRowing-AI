<template>
  <div class="pose-analysis-container">
    <!-- 姿态识别上传区域 -->
    <div class="pose-analysis-section">
      <div class="section-header">
        <h2>训练姿态识别分析</h2>
        <div class="model-selector">
          <el-radio-group v-model="selectedModel" size="small">
            <el-radio-button label="OpenPose">OpenPose模型</el-radio-button>
            <el-radio-button label="MediaPipe">MediaPipe模型</el-radio-button>
            <el-radio-button label="AlphaPose">AlphaPose模型</el-radio-button>
          </el-radio-group>
        </div>
      </div>

      <div class="pose-upload-container">
        <!-- 上传区域 -->
        <div class="upload-area">
          <el-upload
            class="pose-uploader"
            :show-file-list="false"
            :before-upload="handlePoseUpload"
            accept="image/*,video/*"
            drag
          >
            <div class="upload-content">
              <el-icon class="upload-icon" size="48">
                <VideoCameraFilled v-if="!uploadedMedia" />
                <SuccessFilled v-else color="#67c23a" />
              </el-icon>
              <div class="upload-text">
                <p>{{ uploadedMedia ? '文件上传成功' : '拖拽或点击上传训练图片/视频' }}</p>
                <p class="upload-hint">支持 JPG、PNG、MP4 格式，最大50MB</p>
              </div>
            </div>
          </el-upload>
        </div>

        <!-- 分析结果展示 -->
        <div class="analysis-results" v-if="uploadedMedia">
          <div class="media-preview">
            <div class="original-media">
              <h4>原始文件</h4>
              <img v-if="mediaType === 'image'" :src="mediaPreviewUrl" class="media-display" />
              <video v-if="mediaType === 'video'" :src="mediaPreviewUrl" controls class="media-display"></video>
            </div>
            
            <div class="pose-overlay">
              <h4>姿态识别结果 ({{ selectedModel }})</h4>
              <div class="pose-canvas-container" v-loading="poseAnalyzing" element-loading-text="正在分析姿态...">
                <canvas 
                  ref="poseCanvas" 
                  class="pose-canvas"
                  :width="canvasWidth"
                  :height="canvasHeight"
                ></canvas>
                <div class="pose-points-legend">
                  <div class="legend-item">
                    <span class="point-color head"></span>
                    <span>头部关键点</span>
                  </div>
                  <div class="legend-item">
                    <span class="point-color body"></span>
                    <span>躯干关键点</span>
                  </div>
                  <div class="legend-item">
                    <span class="point-color limbs"></span>
                    <span>四肢关键点</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 姿态分析数据 -->
          <div class="pose-metrics">
            <div class="metrics-header">
              <h4>姿态分析指标</h4>
              <div class="metrics-actions">
                <el-button type="info" size="small" @click="resetAnalysis">
                  重新分析
                </el-button>
                <el-button type="primary" size="small" @click="generatePoseReport">
                  生成分析报告
                </el-button>
              </div>
            </div>
            <div class="metrics-grid">
              <div class="metric-card" v-for="metric in poseMetrics" :key="metric.name">
                <div class="metric-icon">
                  <el-icon>
                    <component :is="metric.icon" />
                  </el-icon>
                </div>
                <div class="metric-content">
                  <div class="metric-label">{{ metric.label }}</div>
                  <div class="metric-value" :class="metric.status">{{ metric.value }}</div>
                  <div class="metric-suggestion">{{ metric.suggestion }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- AI分析助手 -->
    <div class="ai-analysis-section">
      <div class="ai-header">
        <h2>AI 姿态分析助手</h2>
        <el-button link @click="clearChat">清空对话</el-button>
      </div>

      <div class="chat-container">
        <el-scrollbar ref="chatScrollbar">
          <div class="chat-messages">
            <div v-for="(msg, index) in chatMessages" :key="index" :class="['message', msg.type]">
              <div class="message-content">
                <template v-if="msg.mediaType">
                  <div class="media-content">
                    <img v-if="msg.mediaType === 'image'" :src="msg.mediaUrl" class="message-image" />
                    <video v-if="msg.mediaType === 'video'" :src="msg.mediaUrl" controls class="message-video"></video>
                  </div>
                </template>
                {{ msg.content }}
              </div>
              <div class="message-time">{{ msg.time }}</div>
            </div>
            <div v-if="isLoading" class="message ai loading-message">
              <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </el-scrollbar>
      </div>

      <div class="chat-input-area">
        <el-upload class="media-upload" :show-file-list="false" :before-upload="handleUpload" accept="image/*,video/*">
          <el-button class="upload-btn" type="primary" text>
            <el-icon>
              <PictureRounded />
            </el-icon>
          </el-button>
        </el-upload>
        <div class="input-wrapper">
          <el-input v-model="chatInput" placeholder="输入问题，按回车发送" @keyup.enter="sendMessage">
            <template #append>
              <el-button type="primary" @click="sendMessage" :loading="isLoading">
                发送
              </el-button>
            </template>
          </el-input>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/stores/user'
import axios from 'axios'
import { 
  PictureRounded, 
  VideoCameraFilled, 
  SuccessFilled,
  User,
  Position,
  TrendCharts,
  Clock
} from '@element-plus/icons-vue'

// 获取用户状态
const userStore = useUserStore()

// 姿态识别相关状态
const selectedModel = ref('OpenPose')
const uploadedMedia = ref(null)
const mediaType = ref('')
const mediaPreviewUrl = ref('')
const poseCanvas = ref(null)
const canvasWidth = ref(400)
const canvasHeight = ref(300)
const poseAnalyzing = ref(false)

// AI聊天相关状态
const chatMessages = ref([])
const chatInput = ref('')
const chatScrollbar = ref(null)
const isLoading = ref(false)
const previewUrl = ref('')
const previewType = ref('')

// 模拟的姿态分析指标
const poseMetrics = ref([
  {
    name: 'posture',
    label: '身体姿态',
    value: '良好',
    status: 'good',
    suggestion: '保持现有姿态',
    icon: 'User'
  },
  {
    name: 'balance',
    label: '身体平衡',
    value: '需改进',
    status: 'warning',
    suggestion: '注意重心稳定',
    icon: 'Position'
  },
  {
    name: 'movement',
    label: '动作流畅性',
    value: '优秀',
    status: 'excellent',
    suggestion: '动作很标准',
    icon: 'TrendCharts'
  },
  {
    name: 'timing',
    label: '动作节奏',
    value: '偏快',
    status: 'warning',
    suggestion: '适当放慢节奏',
    icon: 'Clock'
  }
])

// 模拟关键点数据
const mockKeypoints = [
  // 头部关键点
  { x: 200, y: 50, type: 'head', confidence: 0.9 },
  // 躯干关键点
  { x: 200, y: 120, type: 'body', confidence: 0.95 },
  { x: 180, y: 100, type: 'body', confidence: 0.88 },
  { x: 220, y: 100, type: 'body', confidence: 0.92 },
  { x: 200, y: 180, type: 'body', confidence: 0.87 },
  // 四肢关键点
  { x: 150, y: 140, type: 'limbs', confidence: 0.85 },
  { x: 250, y: 140, type: 'limbs', confidence: 0.89 },
  { x: 120, y: 200, type: 'limbs', confidence: 0.82 },
  { x: 280, y: 200, type: 'limbs', confidence: 0.86 },
  { x: 180, y: 250, type: 'limbs', confidence: 0.91 },
  { x: 220, y: 250, type: 'limbs', confidence: 0.88 }
]

// 姿态识别相关方法
const handlePoseUpload = async (file) => {
  try {
    const isImage = file.type.startsWith('image/')
    const isVideo = file.type.startsWith('video/')

    if (!isImage && !isVideo) {
      ElMessage.error('只支持图片或视频文件')
      return false
    }

    const maxSize = 50 * 1024 * 1024 // 50MB
    if (file.size > maxSize) {
      ElMessage.error('文件大小不能超过50MB')
      return false
    }

    uploadedMedia.value = file
    mediaType.value = isImage ? 'image' : 'video'
    
    // 清理之前的预览URL
    if (mediaPreviewUrl.value) {
      URL.revokeObjectURL(mediaPreviewUrl.value)
    }
    mediaPreviewUrl.value = URL.createObjectURL(file)

    // 延迟执行以确保 DOM 更新
    await nextTick()
    
    // 开始姿态识别处理
    poseAnalyzing.value = true
    setTimeout(() => {
      drawPoseKeypoints()
      poseAnalyzing.value = false
      ElMessage.success(`使用 ${selectedModel.value} 模型识别完成`)
      
      // 自动生成初始分析报告
      generateInitialReport()
    }, 2000)

  } catch (error) {
    console.error('姿态识别上传失败:', error)
    ElMessage.error('文件处理失败，请重试')
    poseAnalyzing.value = false
  }
  return false
}

const drawPoseKeypoints = () => {
  if (!poseCanvas.value) return

  const canvas = poseCanvas.value
  const ctx = canvas.getContext('2d')
  
  // 清空画布
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  
  // 绘制关键点
  mockKeypoints.forEach(point => {
    ctx.beginPath()
    ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI)
    
    // 根据关键点类型设置颜色
    switch (point.type) {
      case 'head':
        ctx.fillStyle = '#ff6b6b'
        break
      case 'body':
        ctx.fillStyle = '#4ecdc4'
        break
      case 'limbs':
        ctx.fillStyle = '#45b7d1'
        break
      default:
        ctx.fillStyle = '#999'
    }
    
    ctx.fill()
    
    // 绘制置信度边框
    if (point.confidence > 0.8) {
      ctx.strokeStyle = '#67c23a'
      ctx.lineWidth = 2
      ctx.stroke()
    } else if (point.confidence > 0.6) {
      ctx.strokeStyle = '#e6a23c'
      ctx.lineWidth = 2
      ctx.stroke()
    } else {
      ctx.strokeStyle = '#f56c6c'
      ctx.lineWidth = 2
      ctx.stroke()
    }
  })
  
  // 绘制骨骼连接线
  const connections = [
    [0, 1], [1, 2], [1, 3], [1, 4], // 头部到躯干
    [4, 5], [4, 6], // 躯干到手臂
    [5, 7], [6, 8], // 手臂延伸
    [4, 9], [4, 10] // 躯干到腿部
  ]
  
  ctx.strokeStyle = '#666'
  ctx.lineWidth = 2
  connections.forEach(([startIdx, endIdx]) => {
    if (mockKeypoints[startIdx] && mockKeypoints[endIdx]) {
      ctx.beginPath()
      ctx.moveTo(mockKeypoints[startIdx].x, mockKeypoints[startIdx].y)
      ctx.lineTo(mockKeypoints[endIdx].x, mockKeypoints[endIdx].y)
      ctx.stroke()
    }
  })
}

const generateInitialReport = () => {
  const initialReport = `🎯 姿态识别分析完成！

检测模型：${selectedModel.value}
识别置信度：${(mockKeypoints.reduce((sum, p) => sum + p.confidence, 0) / mockKeypoints.length * 100).toFixed(1)}%

📊 快速评估：
✅ 检测到 ${mockKeypoints.length} 个关键点
${poseMetrics.value.map(m => `${m.status === 'good' || m.status === 'excellent' ? '✅' : '⚠️'} ${m.label}：${m.value}`).join('\n')}

💡 如需详细分析报告，请点击"生成分析报告"按钮。`

  chatMessages.value.push({
    content: initialReport,
    type: 'ai',
    time: new Date().toLocaleTimeString()
  })
  scrollToBottom()
}

const generatePoseReport = async () => {
  try {
    const reportContent = `
基于 ${selectedModel.value} 模型的详细姿态分析报告：

🎯 整体评估：训练姿态良好，有改进空间

📊 关键指标分析：
• 身体姿态：${poseMetrics.value[0].value} - ${poseMetrics.value[0].suggestion}
• 身体平衡：${poseMetrics.value[1].value} - ${poseMetrics.value[1].suggestion}
• 动作流畅性：${poseMetrics.value[2].value} - ${poseMetrics.value[2].suggestion}
• 动作节奏：${poseMetrics.value[3].value} - ${poseMetrics.value[3].suggestion}

💡 专业改进建议：
1. 注意保持身体重心稳定，避免左右摇摆
2. 适当放慢动作节奏，确保每个动作到位
3. 继续保持良好的身体姿态和动作流畅性
4. 建议加强核心力量训练以提升稳定性

🔍 技术细节：
• 检测到 ${mockKeypoints.length} 个关键点
• 平均置信度：${(mockKeypoints.reduce((sum, p) => sum + p.confidence, 0) / mockKeypoints.length * 100).toFixed(1)}%
• 高置信度点位：${mockKeypoints.filter(p => p.confidence > 0.9).length} 个
• 推荐继续练习以提升动作标准化程度

📈 训练建议：
• 建议每周进行2-3次姿态纠正训练
• 可结合镜子练习，实时观察动作
• 定期使用本系统进行姿态检测和分析
    `

    chatMessages.value.push({
      content: reportContent,
      type: 'ai',
      time: new Date().toLocaleTimeString()
    })
    scrollToBottom()
    ElMessage.success('详细分析报告已生成')
  } catch (error) {
    console.error('生成报告失败:', error)
    ElMessage.error('生成报告失败，请重试')
  }
}

const resetAnalysis = () => {
  if (mediaPreviewUrl.value) {
    URL.revokeObjectURL(mediaPreviewUrl.value)
  }
  uploadedMedia.value = null
  mediaType.value = ''
  mediaPreviewUrl.value = ''
  poseAnalyzing.value = false
  
  // 清空画布
  if (poseCanvas.value) {
    const ctx = poseCanvas.value.getContext('2d')
    ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)
  }
  
  ElMessage.info('已重置，请重新上传文件进行分析')
}

// AI聊天相关方法
const handleUpload = async (file) => {
  try {
    const isImage = file.type.startsWith('image/')
    const isVideo = file.type.startsWith('video/')

    if (!isImage && !isVideo) {
      ElMessage.error('只支持图片或视频文件')
      return false
    }

    previewUrl.value = URL.createObjectURL(file)
    previewType.value = isImage ? 'image' : 'video'

    chatMessages.value.push({
      content: '',
      type: 'user',
      time: new Date().toLocaleTimeString(),
      mediaType: previewType.value,
      mediaUrl: previewUrl.value
    })
    scrollToBottom()

    await sendMessage(true)
  } catch (error) {
    console.error('文件预览失败:', error)
    ElMessage.error('文件预览失败，请重试')
  }
  return false
}

const sendMessage = async (isMediaAnalysis = false) => {
  if ((!chatInput.value.trim() && !isMediaAnalysis) || isLoading.value) return

  if (!isMediaAnalysis) {
    chatMessages.value.push({
      content: chatInput.value,
      type: 'user',
      time: new Date().toLocaleTimeString()
    })
  }

  let userQuestion = ''
  if (isMediaAnalysis) {
    userQuestion = '作为一名专业的赛艇教练，请对以下赛艇训练姿势进行分析和指导：\n\n' +
      '训练者目前的划桨姿势：\n' +
      '1. 起划阶段：\n' +
      '   - 身体前倾约45度\n' +
      '   - 手臂完全伸直\n' +
      '   - 小腿略微前倾\n' +
      '   - 脚掌紧贴踏板\n\n' +
      '2. 驱动阶段：\n' +
      '   - 腿部用力蹬伸\n' +
      '   - 手臂开始弯曲时腿还未完全伸直\n' +
      '   - 身体后仰约30度\n' +
      '   - 划桨高度保持在胸部位置\n\n' +
      '3. 收桨阶段：\n' +
      '   - 手臂收至腹部位置\n' +
      '   - 上身后仰\n' +
      '   - 收放比例约为1:1.5\n' +
      '   - 动作节奏较快\n\n' +
      '请从专业角度分析这些动作要点，指出存在的问题，并给出具体的改进建议。'
  } else {
    userQuestion = chatInput.value
  }
  chatInput.value = ''

  try {
    isLoading.value = true
    
    const response = await axios.post('/ai/chat', {
      user_id: userStore.userInfo.id,
      question: userQuestion
    })

    if (!response.data.success) {
      throw new Error(response.data.message || '获取AI回复失败')
    }

    chatMessages.value.push({
      content: response.data.data,
      type: 'ai',
      time: new Date().toLocaleTimeString()
    })
    scrollToBottom()

    if (previewUrl.value) {
      URL.revokeObjectURL(previewUrl.value)
      previewUrl.value = ''
      previewType.value = ''
    }
  } catch (error) {
    console.error('发送消息失败:', error)
    ElMessage.error('发送消息失败，请稍后重试')
  } finally {
    isLoading.value = false
  }
}

const scrollToBottom = () => {
  setTimeout(() => {
    const scrollbar = chatScrollbar.value
    if (scrollbar) {
      scrollbar.setScrollTop(scrollbar.wrapRef.scrollHeight)
    }
  }, 100)
}

const clearChat = () => {
  chatMessages.value = []
}

onMounted(() => {
  chatMessages.value = [
    {
      content: "你好！我是你的AI姿态分析助手。请上传训练图片或视频，我将为你提供专业的姿态分析和改进建议。",
      type: "ai",
      time: new Date().toLocaleTimeString()
    }
  ]
})

onUnmounted(() => {
  // 清理媒体预览URL
  if (mediaPreviewUrl.value) {
    URL.revokeObjectURL(mediaPreviewUrl.value)
  }
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
})
</script>

<style scoped>
.pose-analysis-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
  background: #f0f2f5;
  min-height: 100vh;
}

/* 姿态识别分析区域样式 */
.pose-analysis-section {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.section-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.model-selector {
  display: flex;
  align-items: center;
  gap: 12px;
}

.pose-upload-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 上传区域样式 */
.upload-area {
  display: flex;
  justify-content: center;
}

.pose-uploader {
  width: 100%;
  max-width: 600px;
}

.pose-uploader :deep(.el-upload) {
  width: 100%;
}

.pose-uploader :deep(.el-upload-dragger) {
  width: 100%;
  height: 200px;
  border: 2px dashed #dcdfe6;
  border-radius: 12px;
  background: #fafafa;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.pose-uploader :deep(.el-upload-dragger:hover) {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  text-align: center;
}

.upload-icon {
  color: #c0c4cc;
  transition: color 0.3s ease;
}

.pose-uploader :deep(.el-upload-dragger:hover) .upload-icon {
  color: var(--el-color-primary);
}

.upload-text p {
  margin: 0;
  font-size: 16px;
  color: #606266;
}

.upload-hint {
  font-size: 14px !important;
  color: #909399 !important;
}

/* 分析结果展示样式 */
.analysis-results {
  display: flex;
  flex-direction: column;
  gap: 24px;
  margin-top: 24px;
}

.media-preview {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

.original-media,
.pose-overlay {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #ebeef5;
}

.original-media h4,
.pose-overlay h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  text-align: center;
}

.media-display {
  width: 100%;
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  object-fit: cover;
  max-height: 300px;
}

.pose-canvas-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  align-items: center;
}

.pose-canvas {
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.pose-points-legend {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #606266;
}

.point-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid #fff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.point-color.head {
  background: #ff6b6b;
}

.point-color.body {
  background: #4ecdc4;
}

.point-color.limbs {
  background: #45b7d1;
}

/* 姿态分析指标样式 */
.pose-metrics {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #ebeef5;
}

.metrics-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.metrics-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.metrics-actions {
  display: flex;
  gap: 8px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.metric-card {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  border: 1px solid #ebeef5;
  display: flex;
  gap: 12px;
  align-items: flex-start;
  transition: all 0.3s ease;
}

.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.metric-icon {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: var(--el-color-primary-light-9);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--el-color-primary);
  flex-shrink: 0;
}

.metric-content {
  flex: 1;
  min-width: 0;
}

.metric-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 4px;
}

.metric-value.good {
  color: #67c23a;
}

.metric-value.warning {
  color: #e6a23c;
}

.metric-value.excellent {
  color: #409eff;
}

.metric-suggestion {
  font-size: 12px;
  color: #606266;
  line-height: 1.4;
}

/* AI分析助手样式 */
.ai-analysis-section {
  background: #fff;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  height: 600px;
}

.ai-header {
  padding: 16px;
  border-bottom: 1px solid #ebeef5;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.ai-header h2 {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.chat-container {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.message {
  max-width: 85%;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.message.user {
  margin-left: auto;
  align-items: flex-end;
}

.message.ai {
  margin-right: auto;
  align-items: flex-start;
}

.message-content {
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  position: relative;
  white-space: pre-wrap;
}

.message.user .message-content {
  background: var(--el-color-primary);
  color: #fff;
  border-bottom-right-radius: 4px;
}

.message.ai .message-content {
  background: #f5f7fa;
  color: #303133;
  border-bottom-left-radius: 4px;
}

.message-time {
  font-size: 12px;
  color: #909399;
}

.chat-input-area {
  padding: 12px;
  border-top: 1px solid #ebeef5;
  background: #fff;
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.media-upload {
  flex-shrink: 0;
}

.input-wrapper {
  flex: 1;
}

.input-wrapper :deep(.el-input__wrapper) {
  box-shadow: none;
  border: 1px solid #dcdfe6;
}

.input-wrapper :deep(.el-input__wrapper):hover {
  border-color: var(--el-color-primary);
}

.input-wrapper :deep(.el-input__wrapper.is-focus) {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 1px var(--el-color-primary) inset;
}

.upload-btn {
  height: 32px;
  width: 32px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #dcdfe6;
  transition: all 0.3s;
}

.upload-btn:hover {
  border-color: var(--el-color-primary);
  color: var(--el-color-primary);
}

.typing-indicator {
  display: inline-flex;
  gap: 4px;
  padding: 8px 12px;
  background: #f5f7fa;
  border-radius: 12px;
  border-bottom-left-radius: 4px;
}

.typing-indicator span {
  width: 6px;
  height: 6px;
  background-color: #909399;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.typing-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%,
  80%,
  100% {
    transform: scale(0);
  }

  40% {
    transform: scale(1);
  }
}

/* 媒体内容样式 */
.media-content {
  margin-bottom: 8px;
}

.message-image {
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
}

.message-video {
  max-width: 240px;
  max-height: 180px;
  border-radius: 8px;
}

/* 响应式布局 */
@media (max-width: 1400px) {
  .media-preview {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .pose-analysis-container {
    padding: 16px;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
  }

  .media-preview {
    grid-template-columns: 1fr;
  }

  .section-header {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }

  .section-header h2 {
    text-align: center;
  }

  .model-selector {
    justify-content: center;
  }

  .pose-uploader :deep(.el-upload-dragger) {
    height: 150px;
  }

  .upload-content {
    gap: 12px;
  }

  .upload-text p {
    font-size: 14px;
  }

  .metrics-header {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }

  .metrics-header h4 {
    text-align: center;
  }

  .metrics-actions {
    justify-content: center;
  }

  .ai-analysis-section {
    height: 500px;
  }
}

@media (max-width: 480px) {
  .pose-analysis-container {
    padding: 12px;
  }

  .pose-analysis-section {
    padding: 16px;
  }

  .model-selector {
    overflow-x: auto;
  }

  .model-selector :deep(.el-radio-group) {
    flex-wrap: nowrap;
    min-width: max-content;
  }

  .pose-points-legend {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>