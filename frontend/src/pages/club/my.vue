<template>
  <div class="my-clubs">
    <el-tabs v-model="activeTab">
      <el-tab-pane label="我管理的俱乐部" name="managed">
        <el-table :data="managedClubs" v-loading="loading">
          <el-table-column prop="name" label="俱乐部名称" />
          <el-table-column prop="description" label="描述" />

          <el-table-column label="操作">
            <template #default="{ row }">
              <el-button link @click="goToDetail(row.clubId)">管理</el-button>
              <el-button link @click="goToActivity(row.clubId)">活动</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
      <el-tab-pane label="我加入的俱乐部" name="joined">
        <el-table :data="joinedClubs" v-loading="loading">
          <el-table-column prop="name" label="俱乐部名称" />
          <el-table-column prop="description" label="描述" />
          <el-table-column prop="joinTime" label="加入时间" />
          <el-table-column label="操作">
            <template #default="{ row }">
              <el-button link @click="goToDetail(row.clubId)">查看</el-button>
              <el-button link type="danger" @click="handleLeave(row.clubId)">
                退出
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>
      <el-tab-pane label="加入申请" name="requests">
        <el-table :data="requestList" v-loading="requestsLoading">
          <el-table-column prop="clubName" label="俱乐部名称" />
          <el-table-column prop="requestTime" label="申请时间" />
          <el-table-column prop="status" label="状态">
            <template #default="{ row }">
              <el-tag :type="getStatusType(row.status)">
                {{ getStatusText(row.status) }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
        
        <el-pagination
          v-model:current-page="requestPage"
          v-model:page-size="requestSize"
          :total="requestTotal"
          @current-change="loadRequests"
        />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { 
  getClubListService, 
  leaveClubService, 
  getJoinRequestListService 
} from '@/api/club'
import { ElMessage, ElMessageBox } from 'element-plus'

const router = useRouter()
const loading = ref(false)
const activeTab = ref('managed')
const managedClubs = ref([])
const joinedClubs = ref([])

// 加载管理的俱乐部
const loadManagedClubs = async () => {
  loading.value = true
  try {
    const res = await getClubListService({
      page: 1,
      size: 100,
      type: 'managed'  // 获取管理的俱乐部
    })
    managedClubs.value = res.list || []
  } catch (error) {
    ElMessage.error('获取管理的俱乐部失败')
  } finally {
    loading.value = false
  }
}

// 加载加入的俱乐部
const loadJoinedClubs = async () => {
  loading.value = true
  try {
    const res = await getClubListService({
      page: 1,
      size: 100,
      type: 'joined'  // 获取加入的俱乐部
    })
    joinedClubs.value = res.list || []
  } catch (error) {
    ElMessage.error('获取加入的俱乐部失败')
  } finally {
    loading.value = false
  }
}

// 退出俱乐部
const handleLeave = async (clubId) => {
  try {
    await ElMessageBox.confirm('确定要退出该俱乐部吗？')
    await leaveClubService({ clubId })
    ElMessage.success('退出成功')
    loadJoinedClubs()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('退出失败')
    }
  }
}

// 页面跳转
const goToDetail = (clubId) => {
  router.push(`/main/club/detail/${clubId}`)
}

const goToActivity = (clubId) => {
  router.push(`/main/club/activity/${clubId}`)
}

// 加入申请相关的状态
const requestsLoading = ref(false)
const requestList = ref([])
const requestPage = ref(1)
const requestSize = ref(10)
const requestTotal = ref(0)

// 加载申请列表
const loadRequests = async () => {
  requestsLoading.value = true
  try {
    const res = await getJoinRequestListService({
      page: requestPage.value,
      size: requestSize.value,
      status: 'all'  // 获取所有状态的申请
    })
    requestList.value = res.list || []
    requestTotal.value = res.total || 0
  } catch (error) {
    ElMessage.error('获取申请列表失败')
  } finally {
    requestsLoading.value = false
  }
}

// 状态显示相关函数
const getStatusType = (status) => {
  const types = {
    0: 'info',    // 待审核
    1: 'success', // 已通过
    2: 'danger'   // 已拒绝
  }
  return types[status] || 'info'
}

const getStatusText = (status) => {
  const texts = {
    0: '待审核',
    1: '已通过',
    2: '已拒绝'
  }
  return texts[status] || '未知'
}

// 在 activeTab 变化时加载对应数据
watch(activeTab, (newVal) => {
  if (newVal === 'managed') {
    loadManagedClubs()
  } else if (newVal === 'joined') {
    loadJoinedClubs()
  } else if (newVal === 'requests') {
    loadRequests()
  }
})

onMounted(() => {
  // 根据当前激活的标签页加载数据
  if (activeTab.value === 'managed') {
    loadManagedClubs()
  } else if (activeTab.value === 'joined') {
    loadJoinedClubs()
  } else if (activeTab.value === 'requests') {
    loadRequests()
  }
})
</script>

<style scoped>
.my-clubs {
  padding: 20px;
}

.el-pagination {
  margin-top: 20px;
  justify-content: flex-end;
}
</style> 