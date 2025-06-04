<template>
  <div class="page-container">
    <!-- 用户基本信息卡片 -->
    <div class="user-profile-section">
      <el-card class="profile-card" shadow="hover">
        <div class="profile-header">
          <div class="profile-avatar">
            <el-image 
              :src="userInfo.avatar || defaultAvatar" 
              fit="cover"
              class="avatar-image" />
          </div>
          <div class="profile-info">
            <h1 class="user-name">{{ userInfo.nickName || '未知用户' }}</h1>
            <p class="user-bio">{{ userInfo.introduction || '这个人很懒，什么都没写~' }}</p>
            <div class="user-stats">
              <div class="stat-item">
                <span class="stat-value">{{ userStats.followingTotal }}</span>
                <span class="stat-label">关注</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ userStats.fansTotal }}</span>
                <span class="stat-label">粉丝</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ userStats.likeTotal }}</span>
                <span class="stat-label">获赞</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ userStats.noteTotal }}</span>
                <span class="stat-label">帖子</span>
              </div>
            </div>
            <el-button v-if="!isSelf" 
                       :type="isFollowing ? 'default' : 'primary'"
                       class="follow-btn"
                       @click="toggleFollow">
              {{ isFollowing ? '已关注' : '关注' }}
            </el-button>
          </div>
        </div>
      </el-card>
    </div>

    <!-- 帖子展示区域 -->
    <div class="content-section">
      <el-card class="content-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <h2>发布的帖子</h2>
            <el-radio-group v-model="viewMode" size="small">
              <el-radio-button label="grid">网格</el-radio-button>
              <el-radio-button label="list">列表</el-radio-button>
            </el-radio-group>
          </div>
        </template>

        <!-- 网格视图 -->
        <div v-if="viewMode === 'grid'" class="notes-grid" v-loading="loading">
          <div v-for="note in notes" 
               :key="note.id" 
               class="note-card"
               @click="goToNoteDetail(note.id)">
            <div class="note-images">
              <img src="/assets/developer.png" alt="默认图片" class="note-image" />
              <img v-if="note.imgUris?.length > 0" 
                   :src="note.imgUris[0]" 
                   :alt="note.title" 
                   class="note-image" 
                   style="position: absolute; top: 0; left: 0;"
                   @error="handleImageError" />
            </div>
            <div class="note-info">
              <h3 class="note-title">{{ note.title }}</h3>
              <p class="note-excerpt">{{ note.content }}</p>
              <div class="note-meta">
                <span class="update-time">
                  <el-icon><Calendar /></el-icon>
                  {{ formatDate(note.createTime || note.updateTime) }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- 列表视图 -->
        <div v-else class="notes-list" v-loading="loading">
          <el-table :data="notes" style="width: 100%">
            <el-table-column prop="title" label="标题" min-width="200">
              <template #default="{ row }">
                <div class="table-title" @click="goToNoteDetail(row.id)">{{ row.title }}</div>
              </template>
            </el-table-column>
            <el-table-column prop="content" label="内容" show-overflow-tooltip />
            <el-table-column prop="updateTime" label="更新时间" width="180">
              <template #default="{ row }">
                {{ formatDate(row.updateTime) }}
              </template>
            </el-table-column>
          </el-table>
        </div>

        <!-- 分页器 -->
        <div class="pagination-container">
          <el-pagination
            background
            layout="prev, pager, next"
            :total="totalNotes"
            :page-size="size"
            v-model:current-page="page"
            @current-change="fetchUserNotes"
          />
        </div>
      </el-card>
    </div>
  </div>
</template>

<style scoped>
.page-container {
  padding: 20px;
  background-color: var(--el-bg-color-page);
}

.user-profile-section {
  margin-bottom: 24px;
}

.profile-card {
  border-radius: 8px;
}

.profile-header {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.profile-avatar {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  overflow: hidden;
  flex-shrink: 0;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.avatar-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.profile-info {
  flex: 1;
}

.user-name {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--el-text-color-primary);
}

.user-bio {
  color: var(--el-text-color-secondary);
  margin-bottom: 16px;
  line-height: 1.5;
}

.user-stats {
  display: flex;
  gap: 32px;
  margin-bottom: 16px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.stat-label {
  font-size: 14px;
  color: var(--el-text-color-secondary);
}

.follow-btn {
  min-width: 100px;
}

.content-card {
  border-radius: 8px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* 网格视图样式 */
.notes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin-top: 16px;
}

.note-card {
  border-radius: 8px;
  overflow: hidden;
  background: white;
  box-shadow: var(--el-box-shadow-light);
  transition: all 0.3s ease;
  cursor: pointer;
  display: flex;
  flex-direction: column;
}

.note-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--el-box-shadow);
}

.note-images {
  position: relative;
  width: 100%;
  height: 200px;
  overflow: hidden;
  border-radius: 8px 8px 0 0;
  background-color: var(--el-fill-color-lighter);
}

.note-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background-color: var(--el-fill-color-lighter);
}

.note-info {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.note-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--el-text-color-primary);
}

.note-excerpt {
  font-size: 14px;
  color: var(--el-text-color-secondary);
  margin-bottom: 12px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.note-meta {
  display: flex;
  align-items: center;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.update-time {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}

.update-time .el-icon {
  font-size: 14px;
}

/* 列表视图样式 */
.table-title {
  color: var(--el-color-primary);
  cursor: pointer;
}

.table-title:hover {
  text-decoration: underline;
}

.pagination-container {
  margin-top: 24px;
  display: flex;
  justify-content: center;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .profile-header {
    flex-direction: column;
    align-items: center;
    text-align: center;
  }

  .user-stats {
    justify-content: center;
  }

  .notes-grid {
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  }
}

/* 优化图片加载效果 */
:deep(.el-image) {
  width: 100%;
  height: 100%;
}

:deep(.el-image__inner) {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

:deep(.el-image__error) {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--el-fill-color-lighter);
}

/* 优化图片加载错误时的显示 */
.note-image[src="/assets/developer.png"] {
  opacity: 0.3;
}

/* 优化实际图片显示 */
.note-image:not([src="/assets/developer.png"]) {
  opacity: 1;
  transition: opacity 0.3s ease;
}
</style>

<script setup>
import { ref, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import axios from 'axios';
import { getToken } from '@/composables/cookie';
import { Calendar, Picture } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { followUserApi, unfollowUserApi } from '@/api/admin/user';

const route = useRoute();
const router = useRouter();

const userInfo = ref({});
const userStats = ref({});
const notes = ref([]);
const loading = ref(false);
const page = ref(1);
const size = ref(10);
const isFollowing = ref(false);
const defaultAvatar =
  'https://img.quanxiaoha.com/quanxiaoha/f97361c0429d4bb1bc276ab835843065.jpg';
const isSelf = ref(false);

const recommendedUsers = ref([
  { id: 1, nickName: 'Alice', avatar: '', followers: 120 },
  { id: 2, nickName: 'Bob', avatar: '', followers: 80 },
  { id: 3, nickName: 'Charlie', avatar: '', followers: 150 },
]);

const formatDate = (dateString) => {
  if (!dateString) return '';
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      return '';
    }
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  } catch (error) {
    console.error('日期格式化错误:', error);
    return '';
  }
};

const fetchUserInfo = async () => {
  try {
    const token = getToken();
    const [userResponse, currentUserResponse] = await Promise.all([
      axios.post(
        '/api/user/user/findById',
        { id: route.params.userId },
        { headers: { Authorization: `Bearer ${token}` } }
      ),
      axios.get('/api/user/user/current', {
        headers: { Authorization: `Bearer ${token}` }
      })
    ]);

    if (userResponse.data.success) {
      userInfo.value = userResponse.data.data;
      if (!userInfo.value.avatar) {
        userInfo.value.avatar = defaultAvatar;
      }
    }

    if (currentUserResponse.data.success) {
      isSelf.value = currentUserResponse.data.data.id === Number(route.params.userId);
    }
  } catch (error) {
    console.error('获取用户信息失败:', error);
  }
};

const fetchUserStats = async () => {
  try {
    const token = getToken();
    const response = await axios.post(
      '/api/count/count/user',
      { userId: route.params.userId },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    if (response.data.success) {
      userStats.value = response.data.data;
    }
  } catch (error) {
    console.error('获取用户统计信息失败:', error);
  }
};

const checkFollowStatus = async () => {
  if (isSelf.value) {
    isFollowing.value = false;
    return;
  }

  try {
    const userId = route.params.userId;
    await followUserApi(userId);
    isFollowing.value = true;
  } catch (error) {
    if (error.response?.data?.errorCode === 'RELATION-20004') {
      // 如果返回已关注错误，说明用户已经关注了
      isFollowing.value = true;
    } else if (error.response?.data?.errorCode === 'RELATION-20001') {
      // 如果是试图关注自己
      isSelf.value = true;
      isFollowing.value = false;
    } else {
      // 其他错误说明未关注
      isFollowing.value = false;
    }
  }
};

const toggleFollow = async () => {
  if (isSelf.value) {
    ElMessage({
      type: 'warning',
      message: '无法关注自己'
    });
    return;
  }

  try {
    const userId = route.params.userId;
    const response = isFollowing.value
      ? await unfollowUserApi(userId)
      : await followUserApi(userId);
      
    if (response.success) {
      isFollowing.value = !isFollowing.value;
      // 更新粉丝数量
      await fetchUserStats();
      ElMessage({
        type: 'success',
        message: isFollowing.value ? '关注成功' : '取消关注成功'
      });
    }
  } catch (error) {
    if (error.response?.data?.errorCode === 'RELATION-20004') {
      // 已经关注的情况
      isFollowing.value = true;
      ElMessage({
        type: 'warning',
        message: '您已经关注了该用户'
      });
    } else if (error.response?.data?.errorCode === 'RELATION-20001') {
      // 试图关注自己的情况
      ElMessage({
        type: 'warning',
        message: '无法关注自己'
      });
    } else {
      console.error(isFollowing.value ? '取消关注失败:' : '关注失败:', error);
      ElMessage({
        type: 'error',
        message: error.response?.data?.message || (isFollowing.value ? '取消关注失败' : '关注失败')
      });
    }
  }
};

const fetchUserNotes = async () => {
  loading.value = true;
  try {
    const token = getToken();
    const response = await axios.post(
      '/api/note/note/UserNoteList',
      { userId: route.params.userId, page: page.value, size: size.value, visible: 0 },
      { headers: { Authorization: `Bearer ${token}` } }
    );

    if (response.data.success) {
      const notesData = response.data.data;
      const notesWithDetails = await Promise.all(
        notesData.map(async (note) => {
          try {
            const detailResponse = await axios.post(
              '/api/note/note/detail',
              { id: note.id },
              { headers: { Authorization: `Bearer ${token}` } }
            );
            if (detailResponse.data.success) {
              return {
                ...note,
                ...detailResponse.data.data,
                createTime: detailResponse.data.data.createTime || note.createTime,
                updateTime: detailResponse.data.data.updateTime || note.updateTime
              };
            }
            return note;
          } catch (error) {
            console.error(`获取帖子 ${note.id} 详情失败:`, error);
            return note;
          }
        })
      );
      notes.value = notesWithDetails;
    }
  } catch (error) {
    console.error('获取帖子列表失败:', error);
  } finally {
    loading.value = false;
  }
};

const goToNoteDetail = (noteId) => {
  router.push({ name: 'NoteDetail', params: { id: noteId, userId: route.params.userId } });
};

// 添加图片错误处理方法
const handleImageError = (event) => {
  event.target.src = "/assets/developer.png";
};

onMounted(async () => {
  await fetchUserInfo();
  await fetchUserStats();
  await checkFollowStatus();
  await fetchUserNotes();
});

// 添加视图模式切换
const viewMode = ref('grid')
</script>