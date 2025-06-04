<template>
  <div class="activity-list">
    <div class="header">
      <h2>俱乐部活动</h2>
      <el-button type="primary" @click="showCreateDialog" v-if="isHost">
        发布活动
      </el-button>
    </div>

    <el-table :data="activityList" v-loading="loading">
      <el-table-column prop="title" label="活动标题" />
      <el-table-column prop="startTime" label="开始时间" />
      <el-table-column prop="endTime" label="结束时间" />
      <el-table-column label="操作">
        <template #default="{ row }">
          <el-button link @click="goToDetail(row.activityId)">查看</el-button>
          <el-button 
            link 
            type="primary" 
            @click="handleSignup(row.activityId)"
            v-if="!isHost"
          >
            报名
          </el-button>
          <template v-if="isHost">
            <el-button link @click="showEditDialog(row)">编辑</el-button>
            <el-button 
              link 
              type="danger" 
              @click="handleDelete(row.activityId)"
            >
              删除
            </el-button>
          </template>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-model:current-page="page"
      v-model:page-size="size"
      :total="total"
      @current-change="loadActivities"
    />

    <!-- 创建/编辑活动对话框 -->
    <el-dialog 
      :title="isEdit ? '编辑活动' : '发布活动'" 
      v-model="dialogVisible"
    >
      <el-form :model="form" ref="formRef">
        <el-form-item label="标题" prop="title">
          <el-input v-model="form.title" />
        </el-form-item>
        <el-form-item label="内容" prop="content">
          <el-input type="textarea" v-model="form.content" rows="4" />
        </el-form-item>
        <el-form-item label="时间范围" prop="timeRange">
          <el-date-picker
            v-model="timeRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { 
  getClubActivityListService,
  publishActivityService,
  deleteActivityService,
  getActivityDetailsService,
  updateActivityService,
  getHostListService,
  signupActivityService
} from '@/api/club'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/stores/user'

const route = useRoute()
const router = useRouter()
const clubId = route.params.clubId

const loading = ref(false)
const activityList = ref([])
const page = ref(1)
const size = ref(10)
const total = ref(0)
const isHost = ref(false)

// 表单相关
const dialogVisible = ref(false)
const isEdit = ref(false)
const form = ref({
  title: '',
  content: '',
})
const timeRange = ref([])

// 新增用户store的引入
const userStore = useUserStore()
const currentUserId = computed(() => userStore.userInfo.id)

// 加载活动列表
const loadActivities = async () => {
  loading.value = true
  try {
    const res = await getClubActivityListService({
      clubId,
      page: page.value,
      size: size.value
    })
    activityList.value = res.data
    total.value = res.total
  } catch (error) {
    ElMessage.error('获取活动列表失败')
  }
  loading.value = false
}

// 检查是否是主持人
const checkIsHost = async () => {
  try {
    const res = await getHostListService({ clubId })
    isHost.value = res.data.some(host => host.userId === currentUserId.value)
  } catch (error) {
    console.error(error)
  }
}

// 处理创建/编辑活动
const handleSubmit = async () => {
  if (!timeRange.value || timeRange.value.length !== 2) {
    ElMessage.warning('请选择活动时间范围')
    return
  }

  const [startTime, endTime] = timeRange.value
  const data = {
    ...form.value,
    startTime,
    endTime,
    clubId
  }

  try {
    if (isEdit.value) {
      await updateActivityService(data)
      ElMessage.success('更新成功')
    } else {
      await publishActivityService(data)
      ElMessage.success('发布成功')
    }
    dialogVisible.value = false
    loadActivities()
  } catch (error) {
    ElMessage.error(isEdit.value ? '更新失败' : '发布失败')
  }
}

// 处理删除活动
const handleDelete = async (activityId) => {
  try {
    await ElMessageBox.confirm('确定要删除该活动吗？')
    await deleteActivityService({ activityId, clubId })
    ElMessage.success('删除成功')
    loadActivities()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

// 处理活动报名
const handleSignup = async (activityId) => {
  try {
    await signupActivityService({ activityId })
    ElMessage.success('报名成功')
  } catch (error) {
    ElMessage.error('报名失败')
  }
}

// 显示编辑对话框
const showEditDialog = (activity) => {
  isEdit.value = true
  form.value = {
    activityId: activity.activityId,
    title: activity.title,
    content: activity.content
  }
  timeRange.value = [activity.startTime, activity.endTime]
  dialogVisible.value = true
}

// 显示创建对话框
const showCreateDialog = () => {
  isEdit.value = false
  form.value = {
    title: '',
    content: ''
  }
  timeRange.value = []
  dialogVisible.value = true
}

// 跳转到活动详情
const goToDetail = (activityId) => {
  router.push(`/main/club/activity/${activityId}`)
}

onMounted(() => {
  checkIsHost()
  loadActivities()
})
</script>

<style scoped>
.activity-list {
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
</style> 