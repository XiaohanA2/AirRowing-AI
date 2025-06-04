<template>
  <div class="follow-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>社交关系</h1>
      <p class="subtitle">管理您的关注和粉丝</p>
    </div>

    <!-- 顶部统计卡片 -->
    <div class="stat-cards">
      <el-card class="stat-card" shadow="hover" @click="showFollowings">
        <div class="stat-content">
          <div class="stat-icon following-icon">
            <el-icon><UserFilled /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ followCount }}</div>
            <div class="stat-label">关注</div>
          </div>
        </div>
      </el-card>
      
      <el-card class="stat-card" shadow="hover" @click="showFans">
        <div class="stat-content">
          <div class="stat-icon fans-icon">
            <el-icon><Star /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ fansCount }}</div>
            <div class="stat-label">粉丝</div>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 关注/粉丝列表抽屉 -->
    <el-drawer
      v-model="drawerVisible"
      :title="drawerType === 'following' ? '关注列表' : '粉丝列表'"
      direction="rtl"
      size="80%"
      :destroy-on-close="true"
      class="custom-drawer"
    >
      <template #header>
        <div class="drawer-header">
          <h2>{{ drawerType === 'following' ? '关注列表' : '粉丝列表' }}</h2>
          <div class="drawer-stats">
            <span class="stat-item">
              <el-icon><User /></el-icon>
              总数：{{ drawerType === 'following' ? followCount : fansCount }}
            </span>
          </div>
        </div>
      </template>

      <div class="drawer-content">
        <!-- 搜索和筛选区域 -->
        <div class="search-filter-section">
          <el-input
            v-model="searchQuery"
            placeholder="搜索用户..."
            class="search-input"
            :prefix-icon="Search"
            clearable
          >
            <template #append>
              <el-button :icon="Search" />
            </template>
          </el-input>

          <el-select v-model="sortBy" placeholder="排序方式" class="sort-select">
            <el-option label="最近关注" value="recent" />
            <el-option label="粉丝数量" value="fans" />
            <el-option label="帖子数量" value="notes" />
          </el-select>
        </div>

        <!-- 用户列表 -->
        <div 
          class="user-list" 
          v-infinite-scroll="loadMore" 
          :infinite-scroll-disabled="loading"
          :infinite-scroll-distance="10"
        >
          <el-row :gutter="20">
            <el-col :span="8" v-for="user in filteredUsers" :key="user.userId">
              <el-card class="user-card" shadow="hover">
                <div class="user-card-content">
                  <div class="user-card-header">
                    <el-avatar 
                      :size="64" 
                      :src="user.avatar" 
                      @click="goToUserDetail(user.userId)"
                      class="clickable user-avatar"
                    />
                    <div class="user-basic-info">
                      <h3 class="user-nickname clickable" @click="goToUserDetail(user.userId)">
                        {{ user.nickname }}
                        <el-tag size="small" v-if="user.isActive" type="success">活跃</el-tag>
                      </h3>
                      <p class="user-intro">{{ user.introduction || '这个人很懒，什么都没写~' }}</p>
                    </div>
                  </div>
                  
                  <div class="user-stats">
                    <div class="stat-item">
                      <span class="stat-value">{{ user.userStats?.fansTotal || 0 }}</span>
                      <span class="stat-label">粉丝</span>
                    </div>
                    <div class="stat-item">
                      <span class="stat-value">{{ user.userStats?.noteTotal || 0 }}</span>
                      <span class="stat-label">帖子</span>
                    </div>
                    <div class="stat-item">
                      <span class="stat-value">{{ user.userStats?.likeTotal || 0 }}</span>
                      <span class="stat-label">获赞</span>
                    </div>
                  </div>

                  <div class="user-action">
                    <el-button
                      type="primary"
                      size="small"
                      plain
                      @click="goToUserDetail(user.userId)"
                      class="view-button"
                    >
                      <el-icon><View /></el-icon>
                      <span>查看</span>
                    </el-button>
                  </div>
                </div>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <!-- 加载状态 -->
        <div class="loading-state" v-if="loading">
          <el-skeleton :rows="3" animated />
          <div class="loading-text">加载中...</div>
        </div>

        <!-- 空状态 -->
        <el-empty
          v-if="!loading && filteredUsers.length === 0"
          :description="drawerType === 'following' ? '还没有关注任何人' : '还没有粉丝'"
        >
          <template #image>
            <el-icon class="empty-icon"><UserFilled /></el-icon>
          </template>
          <el-button type="primary" @click="handleEmptyAction">
            {{ drawerType === 'following' ? '去发现用户' : '去互动' }}
          </el-button>
        </el-empty>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { 
  UserFilled, Star, Search, ArrowRight, 
  Connection, Check, View, User 
} from '@element-plus/icons-vue'
import { 
  currentUserService, 
  followService, 
  fansService, 
  noFollowService, 
  doFollowService 
} from '@/api/follow.js'
import axios from 'axios'
import { getToken } from '@/composables/cookie'

// 状态变量
const drawerVisible = ref(false)
const drawerType = ref('following') // 'following' 或 'fans'
const searchQuery = ref('')
const loading = ref(false)
const pageNo = ref(1)
const pageSize = ref(12)
const hasMore = ref(true)

const userId = ref()
const followCount = ref(0)
const fansCount = ref(0)
const userList = ref([])

// 计算过滤后的用户列表
const filteredUsers = computed(() => {
  if (!searchQuery.value) return userList.value
  return userList.value.filter(user => 
    user.nickname.toLowerCase().includes(searchQuery.value.toLowerCase())
  )
})

// 显示关注列表
const showFollowings = () => {
  drawerType.value = 'following'
  drawerVisible.value = true
  resetList()
  loadUsers()
}

// 显示粉丝列表
const showFans = () => {
  drawerType.value = 'fans'
  drawerVisible.value = true
  resetList()
  loadUsers()
}

// 重置列表状态
const resetList = () => {
  userList.value = []
  pageNo.value = 1
  hasMore.value = true
  searchQuery.value = ''
}

// 加载更多用户
const loadMore = async () => {
  if (loading.value || !hasMore.value) return
  await loadUsers()
}

// 加载用户数据
const loadUsers = async () => {
  if (loading.value) return;
  loading.value = true;
  
  try {
    const params = {
      userId: userId.value,
      pageNo: pageNo.value,
      pageSize: pageSize.value
    };
    
    const service = drawerType.value === 'following' ? followService : fansService;
    const result = await service(params);
    
    if (result.success) {
      // 检查是否有数据
      if (!result.data || result.data.length === 0) {
        hasMore.value = false;
        userList.value = [];
        
        // 更新计数
        if (drawerType.value === 'following') {
          followCount.value = result.totalCount || 0;
        } else {
          fansCount.value = result.totalCount || 0;
        }
        return;
      }

      // 获取每个用户的统计信息
      const usersWithStats = await Promise.all(
        result.data.map(async user => {
          const stats = await fetchUserStats(user.userId);
          return {
            ...user,
            loading: false,
            userStats: stats,
            avatar: user.avatar || 'https://img.quanxiaoha.com/quanxiaoha/f97361c0429d4bb1bc276ab835843065.jpg'
          };
        })
      );
      
      userList.value.push(...usersWithStats);
      hasMore.value = usersWithStats.length === pageSize.value;
      pageNo.value++;
      
      // 更新计数
      if (drawerType.value === 'following') {
        followCount.value = result.totalCount || 0;
      } else {
        fansCount.value = result.totalCount || 0;
      }
    } else {
      hasMore.value = false;
    }
  } catch (error) {
    console.error('加载用户失败:', error);
    hasMore.value = false;
  } finally {
    loading.value = false;
  }
};

// 切换关注状态
const toggleFollow = async (user) => {
  if (user.loading) return
  user.loading = true
  
  try {
    if (user.isFollowed) {
      await noFollowService({ unfollowUserId: user.userId })
    } else {
      await doFollowService({ followUserId: user.userId })
    }
    
    user.isFollowed = !user.isFollowed
    ElMessage.success(user.isFollowed ? '关注成功' : '取消关注成功')
    
    // 更新关注数
    const result = await followService({ userId: userId.value, pageNo: 1 })
    if (result.success) {
      followCount.value = result.totalCount
    }
  } catch (error) {
    ElMessage.error('操作失败')
  } finally {
    user.loading = false
  }
}

const router = useRouter()
const goToUserDetail = (userId) => {
  router.push(`/user/${userId}`)
  drawerVisible.value = false
}

// 初始化
const init = async () => {
  try {
    const result = await currentUserService()
    if (result.success) {
      userId.value = result.data.id
      await Promise.all([
        followService({ userId: userId.value, pageNo: 1 }).then(res => {
          if (res.success) followCount.value = res.totalCount
        }),
        fansService({ userId: userId.value, pageNo: 1 }).then(res => {
          if (res.success) fansCount.value = res.totalCount
        })
      ])
    }
  } catch (error) {
    console.error('初始化失败:', error)
  }
}

// 新增排序方式
const sortBy = ref('recent')

// 添加新的 API 调用函数
const fetchUserStats = async (userId) => {
  try {
    const token = getToken()
    const response = await axios.post('/api/count/count/user', 
      { userId },
      { headers: { Authorization: `Bearer ${token}` } }
    )
    if (response.data.success) {
      return response.data.data
    }
    return null
  } catch (error) {
    console.error('获取用户统计信息失败:', error)
    return null
  }
}

// 处理空状态操作
const handleEmptyAction = () => {
  if (drawerType.value === 'following') {
    router.push('/admin/index') // 跳转到推荐页面
  } else {
    ElMessage({
      type: 'info',
      message: '多发布优质内容，增加互动机会吧！'
    })
  }
}

onMounted(() => {
  init()
})
</script>

<style scoped>
.follow-container {
  padding: 30px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 30px;
}

.page-header h1 {
  margin: 0;
  font-size: 28px;
  color: #303133;
}

.subtitle {
  margin: 8px 0 0;
  color: #909399;
  font-size: 14px;
}

.stat-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  margin-bottom: 30px;
}

.stat-card {
  cursor: pointer;
  transition: all 0.3s ease;
  border-radius: 12px;
  overflow: hidden;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.stat-content {
  display: flex;
  align-items: center;
  padding: 24px;
  position: relative;
}

.stat-icon {
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  margin-right: 20px;
  font-size: 24px;
}

.following-icon {
  background-color: #ecf5ff;
  color: #409eff;
}

.fans-icon {
  background-color: #fdf6ec;
  color: #e6a23c;
}

.interaction-icon {
  background-color: #f0f9eb;
  color: #67c23a;
}

.stat-info {
  flex: 1;
}

.stat-number {
  font-size: 28px;
  font-weight: bold;
  color: #303133;
  line-height: 1.2;
}

.stat-label {
  color: #909399;
  margin-top: 8px;
  font-size: 14px;
}

.stat-trend {
  position: absolute;
  right: 24px;
  color: #909399;
  transition: transform 0.3s ease;
}

.stat-card:hover .stat-trend {
  transform: translateX(5px);
  color: #409eff;
}

.custom-drawer :deep(.el-drawer__header) {
  margin-bottom: 0;
  padding: 20px;
  border-bottom: 1px solid #ebeef5;
}

.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.drawer-header h2 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

.drawer-stats {
  display: flex;
  gap: 16px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #606266;
}

.search-filter-section {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
}

.search-input {
  flex: 1;
}

.sort-select {
  width: 140px;
}

.user-list {
  height: calc(100vh - 280px);
  overflow-y: auto;
  padding: 0 4px;
}

.user-card {
  margin-bottom: 20px;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.user-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
}

.user-card-content {
  padding: 20px;
}

.user-card-header {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.user-avatar {
  border: 2px solid #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.user-basic-info {
  flex: 1;
  min-width: 0;
}

.user-nickname {
  margin: 0 0 8px;
  font-size: 16px;
  color: #303133;
  display: flex;
  align-items: center;
  gap: 8px;
}

.user-intro {
  margin: 0;
  font-size: 14px;
  color: #909399;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  line-height: 1.4;
}

.user-stats {
  display: flex;
  justify-content: space-around;
  padding: 16px 0;
  border-top: 1px solid #ebeef5;
  border-bottom: 1px solid #ebeef5;
  margin: 16px 0;
}

.user-stats .stat-item {
  text-align: center;
}

.user-stats .stat-value {
  display: block;
  font-size: 16px;
  font-weight: bold;
  color: #303133;
}

.user-stats .stat-label {
  font-size: 12px;
  color: #909399;
}

.user-action {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.view-button {
  min-width: 90px;
}

.loading-state {
  padding: 20px;
  text-align: center;
}

.loading-text {
  margin-top: 12px;
  color: #909399;
}

.empty-icon {
  font-size: 60px;
  color: #909399;
}

/* 滚动条美化 */
.user-list::-webkit-scrollbar {
  width: 6px;
}

.user-list::-webkit-scrollbar-thumb {
  background-color: #dcdfe6;
  border-radius: 3px;
}

.user-list::-webkit-scrollbar-track {
  background-color: #f5f7fa;
}

@media (max-width: 1200px) {
  .stat-cards {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .stat-cards {
    grid-template-columns: 1fr;
  }
}
</style>
  