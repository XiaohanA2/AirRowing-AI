<template>
  <div class="club-list">
    <div class="page-header">
      <div class="header-left">
        <h2 class="page-title">俱乐部</h2>
        <div class="search-container">
          <el-input
            v-model="searchQuery"
            placeholder="搜索俱乐部名称或描述"
            clearable
            class="search-input"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </div>
      </div>
      <div class="header-right">
        <el-button type="primary" @click="showCreateDialog">
          <el-icon><Plus /></el-icon>
          创建俱乐部
        </el-button>
      </div>
    </div>

    <el-table :data="filteredClubs" v-loading="loading">
      <el-table-column prop="name" label="俱乐部名称" />
      <el-table-column prop="description" label="描述" />
      <el-table-column prop="createTime" label="" />
      <el-table-column label="操作">
        <template #default="{ row }">
          <el-button link @click="goToDetail(row.clubId)">查看</el-button>
          <el-button link @click="handleJoin(row.clubId)">申请加入</el-button>
        </template>
      </el-table-column>
    </el-table>
    
    <el-pagination
      v-model:current-page="page"
      v-model:page-size="size"
      :total="total"
      @current-change="loadClubs"
    />
    
    <!-- 创建俱乐部对话框 -->
    <el-dialog v-model="createDialogVisible" title="创建俱乐部">
      <el-form :model="createForm" ref="createFormRef">
        <el-form-item label="名称" prop="name">
          <el-input v-model="createForm.name" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input type="textarea" v-model="createForm.description" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { 
  createClubService, 
  getClubListService, 
  joinClubService 
} from '@/api/club'
import { ElMessage } from 'element-plus'

const router = useRouter()
const loading = ref(false)
const clubList = ref([])
const page = ref(1)
const size = ref(10)
const total = ref(0)

const createDialogVisible = ref(false)
const createForm = ref({
  name: '',
  description: ''
})

const searchQuery = ref('')

const filteredClubs = computed(() => {
  if (!searchQuery.value) {
    return clubList.value
  }
  const query = searchQuery.value.toLowerCase()
  return clubList.value.filter(club => 
    club.name.toLowerCase().includes(query) ||
    club.description.toLowerCase().includes(query)
  )
})

const loadClubs = async () => {
  loading.value = true
  try {
    console.log('Loading clubs with page:', page.value, 'size:', size.value);
    const res = await getClubListService({
      page: page.value,
      size: size.value
    })
    console.log('Response from server:', res);
    if (res.success) {
      clubList.value = res.data?.clubs || []
      total.value = parseInt(res.data?.total || '0')
      page.value = res.data?.currentPage || 1
      size.value = res.data?.pageSize || 10
    } else {
      ElMessage.error(res.message || '获取俱乐部列表失败')
    }
  } catch (error) {
    console.error('完整错误信息:', error);
    ElMessage.error(error.response?.data?.message || '获取俱乐部列表失败')
  } finally {
    loading.value = false
  }
}

const handleCreate = async () => {
  try {
    if (!createForm.value.name.trim()) {
      ElMessage.warning('俱乐部名称不能为空')
      return
    }
    
    const res = await createClubService(createForm.value)
    if (res.success) {
      ElMessage.success('创建成功')
      createDialogVisible.value = false
      createForm.value = { name: '', description: '' }  // 清空表单
      loadClubs()  // 重新加载列表
    } else {
      ElMessage.error(res.message || '创建失败')
    }
  } catch (error) {
    console.error('创建俱乐部出错:', error)
    ElMessage.error('创建失败')
  }
}

const handleJoin = async (clubId) => {
  try {
    const res = await joinClubService({ clubId })
    if (res.success) {
      ElMessage.success(res.data || '申请已提交')
    } else {
      ElMessage.error(res.message || '申请失败')
    }
  } catch (error) {
    console.error('申请加入俱乐部出错:', error)
    ElMessage.error(error.response?.data?.message || '申请失败')
  }
}

const goToDetail = (clubId) => {
  router.push(`/main/club/detail/${clubId}`)
}

const showCreateDialog = () => {
  createForm.value = { name: '', description: '' }  // 清空表单
  createDialogVisible.value = true
}

onMounted(() => {
  loadClubs()
})
</script>

<style scoped>
.club-list {
  padding: 32px;
  max-width: 1400px;
  margin: 0 auto;
  min-height: calc(100vh - 64px);
  background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  background: rgba(255, 255, 255, 0.9);
  padding: 20px 24px;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 24px;
  flex: 1;
}

.page-title {
  margin: 0;
  font-size: 24px;
  font-weight: 700;
  color: #1a1a1a;
  white-space: nowrap;
}

.search-container {
  flex: 1;
  max-width: 400px;
}

.search-input {
  --el-input-hover-border-color: #409EFF;
  --el-input-focus-border-color: #409EFF;
}

:deep(.el-input__wrapper) {
  padding: 4px 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
  border-radius: 8px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  background: #f5f7fa;
}

:deep(.el-input__wrapper:hover) {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  background: #ffffff;
}

:deep(.el-input__wrapper.is-focus) {
  background: #ffffff;
}

:deep(.el-input__inner) {
  height: 36px;
  font-size: 14px;
}

:deep(.el-input__prefix) {
  color: #909399;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-right .el-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 36px;
  padding: 0 16px;
  font-weight: 600;
  border-radius: 8px;
}

.club-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 24px;
  margin-top: 24px;
}

@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
    padding: 16px;
  }

  .header-left {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }

  .search-container {
    max-width: 100%;
  }

  .header-right {
    justify-content: flex-end;
  }

  .club-list {
    padding: 16px;
  }

  .search-container {
    margin-bottom: 16px;
  }

  .club-grid {
    gap: 16px;
  }
}
</style> 