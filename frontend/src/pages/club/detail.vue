<template>
  <div class="club-detail">
    <!-- 顶部操作栏 -->
    <el-card class="action-card">
      <div class="action-header">
        <div class="club-title">
          <h1>{{ clubInfo.name }}</h1>
          <el-tag 
            v-if="isCreator" 
            type="success" 
            effect="dark"
          >主持人</el-tag>
          <el-tag 
            v-else-if="isMember" 
            type="info" 
            effect="dark"
          >成员</el-tag>
          <!-- 添加创建人信息 -->
          <div v-if="creatorInfo.id" class="creator-info-inline">
            <span class="creator-label">创建人：</span>
            <el-avatar 
              :size="32" 
              :src="creatorInfo.avatar"
              @click="goToUserDetail(creatorInfo.id)"
            />
            <span class="creator-name-inline" @click="goToUserDetail(creatorInfo.id)">
              {{ creatorInfo.nickName }}
            </span>
          </div>
        </div>
        <div class="action-buttons">
          <el-button 
            v-if="isCreator" 
            type="primary" 
            @click="showRequestsDialog"
          >
            <el-icon><UserFilled /></el-icon>
            查看申请列表
          </el-button>
          <el-button type="primary" @click="showMembersDialog">
            <el-icon><User /></el-icon>
            查看成员列表
          </el-button>
          <el-button 
            v-if="isCreator" 
            type="primary" 
            @click="showEditDialog"
          >
            <el-icon><Edit /></el-icon>
            编辑信息
          </el-button>
          <el-button 
            v-if="isCreator" 
            type="success" 
            @click="showPublishActivityDialog"
          >
            <el-icon><Plus /></el-icon>
            发布活动
          </el-button>
          <el-button 
            v-if="!isCreator && isMember" 
            type="danger" 
            @click="handleLeaveClub"
          >
            <el-icon><SwitchButton /></el-icon>
            退出俱乐部
          </el-button>
          <el-button 
            v-if="isCreator" 
            type="danger" 
            @click="handleDeleteClub"
          >
            <el-icon><Delete /></el-icon>
            删除俱乐部
          </el-button>
          <el-button @click="goBack">
            <el-icon><Back /></el-icon>
            返回
          </el-button>
        </div>
      </div>
    </el-card>

    <div class="content-container">
      <!-- 左侧：俱乐部基本信息 -->
      <div class="left-section">
        <el-card class="info-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <h3>俱乐部信息</h3>
            </div>
          </template>
          
          <div class="info-section" v-if="clubInfo.clubId">
            <div class="info-item">
              <label>俱乐部描述：</label>
              <div class="description">{{ clubInfo.description }}</div>
            </div>
            <div class="info-item">
              <label>更新时间：</label>
              <span>{{ formatDate(clubInfo.updateTime) }}</span>
            </div>
          </div>
        </el-card>
      </div>

      <!-- 右侧：活动列表 -->
      <div class="right-section">
        <el-card class="activities-card">
          <template #header>
            <div class="card-header">
              <h3>俱乐部活动</h3>
              <el-button 
                v-if="isCreator"
                type="primary" 
                link 
                @click="showPublishActivityDialog"
              >
                <el-icon><Plus /></el-icon>
                发布新活动
              </el-button>
            </div>
          </template>

          <div class="activities-content" v-loading="activitiesLoading">
            <el-empty 
              v-if="!activities.length" 
              description="暂无活动"
            >
              <template #extra>
                <el-button 
                  v-if="isCreator"
                  type="primary" 
                  @click="showPublishActivityDialog"
                >
                  发布第一个活动
                </el-button>
              </template>
            </el-empty>
            
            <template v-else>
              <div class="activity-grid">
                <div 
                  v-for="activity in activities" 
                  :key="activity.activityId"
                  class="activity-card"
                >
                  <div class="activity-header">
                    <h4>{{ activity.title }}</h4>
                  </div>
                  <div class="activity-meta">
                    <div class="time">
                      <el-icon><Timer /></el-icon>
                      {{ formatDate(activity.startTime) }}
                    </div>
                    <div class="time">
                      <el-icon><Timer /></el-icon>
                      {{ formatDate(activity.endTime) }}
                    </div>
                  </div>
                </div>
              </div>

              <el-pagination
                v-if="activityTotal > 0"
                v-model:current-page="activityPage"
                v-model:page-size="activitySize"
                :total="activityTotal"
                @current-change="loadActivities"
                class="pagination"
                layout="prev, pager, next"
              />
            </template>
          </div>
        </el-card>
      </div>
    </div>

    <!-- 申请列表对话框 -->
    <el-dialog 
      v-model="requestsDialogVisible" 
      title="加入申请列表"
      width="800px"
    >
      <div v-loading="requestsLoading">
        <el-empty v-if="!joinRequests.length" description="暂无申请记录" />
        <template v-else>
          <div class="requests-list">
            <div 
              v-for="request in joinRequests" 
              :key="request.userId"
              class="request-item"
            >
              <el-avatar 
                :size="50" 
                :src="request.avatar"
                @click="goToUserDetail(request.userId)"
              />
              <div class="request-info">
                <div class="request-user">{{ request.nickName }}</div>
                <div class="request-time">申请时间：{{ formatDate(request.createTime) }}</div>
              </div>
              <div class="request-actions" v-if="request.status === 0">
                <el-button 
                  type="primary" 
                  @click="handleReview(request.userId, 1)"
                >
                  同意
                </el-button>
                <el-button 
                  type="danger" 
                  @click="handleReview(request.userId, 2)"
                >
                  拒绝
                </el-button>
              </div>
              <div v-else class="status-text" :class="getStatusType(request.status)">
                {{ getStatusText(request.status) }}
              </div>
            </div>
          </div>

          <el-pagination
            v-if="requestTotal > 0"
            v-model:current-page="requestPage"
            v-model:page-size="requestSize"
            :total="requestTotal"
            @current-change="loadJoinRequests"
            class="pagination"
            layout="prev, pager, next"
          />
        </template>
      </div>
    </el-dialog>

    <!-- 成员列表对话框 -->
    <el-dialog 
      v-model="membersDialogVisible" 
      title="俱乐部成员"
      width="800px"
    >
      <div v-loading="membersLoading">
        <el-empty v-if="!members.length" description="暂无成员" />
        <template v-else>
          <div class="members-grid">
            <div 
              v-for="member in members" 
              :key="member.userId"
              class="member-card"
            >
              <el-avatar 
                :size="60" 
                :src="member.avatar"
                @click="goToUserDetail(member.userId)"
              />
              <div class="member-info">
                <div class="member-name">{{ member.nickname }}</div>
                <div class="member-join-time">加入时间：{{ member.joinTime }}</div>
                <div class="member-actions">
                  <el-button 
                    link 
                    type="primary"
                    @click="goToUserDetail(member.userId)"
                  >
                    查看资料
                  </el-button>
                  <el-button 
                    v-if="isCreator && String(member.userId) !== String(clubInfo.creatorId)"
                    link 
                    type="danger"
                    @click="handleRemoveMember(member.userId)"
                  >
                    移除
                  </el-button>
                </div>
              </div>
            </div>
          </div>

          <el-pagination
            v-if="memberTotal > 0"
            v-model:current-page="memberPage"
            v-model:page-size="memberSize"
            :total="memberTotal"
            @current-change="loadMembers"
            class="pagination"
            layout="prev, pager, next"
          />
        </template>
      </div>
    </el-dialog>

    <!-- 编辑信息对话框 -->
    <el-dialog 
      v-model="editDialogVisible" 
      title="编辑俱乐部信息"
      width="500px"
    >
      <el-form 
        ref="editFormRef"
        :model="editForm"
        :rules="editRules"
        label-width="100px"
        class="edit-form"
      >
        <el-form-item label="俱乐部名称" prop="name">
          <el-input v-model="editForm.name" placeholder="请输入俱乐部名称" />
        </el-form-item>
        <el-form-item label="俱乐部描述" prop="description">
          <el-input 
            v-model="editForm.description" 
            type="textarea" 
            :rows="4"
            placeholder="请输入俱乐部描述"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="editDialogVisible = false">取消</el-button>
          <el-button type="primary" @click="handleUpdateClub" :loading="updating">
            确定
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- 发布活动对话框 -->
    <el-dialog 
      v-model="activityDialogVisible" 
      title="发布活动"
      width="600px"
    >
      <el-form 
        ref="activityFormRef"
        :model="activityForm"
        :rules="activityRules"
        label-width="100px"
        class="publish-form"
      >
        <el-form-item label="活动标题" prop="title">
          <el-input v-model="activityForm.title" placeholder="请输入活动标题" />
        </el-form-item>
        <el-form-item label="活动内容" prop="content">
          <el-input 
            v-model="activityForm.content" 
            type="textarea" 
            :rows="6"
            placeholder="请输入活动内容"
          />
        </el-form-item>
        <el-form-item label="活动时间" prop="time">
          <div class="time-picker-wrapper">
            <el-date-picker
              v-model="activityForm.time"
              type="datetimerange"
              range-separator="至"
              start-placeholder="开始时间"
              end-placeholder="结束时间"
              value-format="YYYY-MM-DDTHH:mm:ss"
              :default-time="[
                new Date(2000, 1, 1, 9, 0, 0),
                new Date(2000, 1, 1, 17, 0, 0)
              ]"
            />
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="activityDialogVisible = false">取消</el-button>
          <el-button 
            type="primary" 
            @click="handlePublishActivity" 
            :loading="publishing"
          >
            发布
          </el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { 
  getClubDetailsService, 
  getClubJoinRequestsService, 
  reviewJoinRequestService,
  getClubMembersService,
  leaveClubService,
  removeMemberService,
  deleteClubService,
  updateClubService,
  publishActivityService,
  getClubActivityListService
} from '@/api/club'
import { getUserInfoByIdService } from '@/api/user'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/stores/user'

// 添加日期格式化函数
const formatDate = (dateString) => {
  if (!dateString) return '暂无数据'
  const date = new Date(dateString)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  })
}

const route = useRoute()
const router = useRouter()
const userStore = useUserStore()
const loading = ref(false)
const clubInfo = ref({})
const creatorInfo = ref({})

// 加入申请列表相关
const requestsLoading = ref(false)
const joinRequests = ref([])
const requestPage = ref(1)
const requestSize = ref(10)
const requestTotal = ref(0)
const requestsDialogVisible = ref(false)

// 成员列表相关
const membersDialogVisible = ref(false)
const membersLoading = ref(false)
const members = ref([])
const memberPage = ref(1)
const memberSize = ref(10)
const memberTotal = ref(0)

// 编辑相关
const editDialogVisible = ref(false)
const editFormRef = ref(null)
const updating = ref(false)
const editForm = ref({
  name: '',
  description: ''
})
const editRules = {
  name: [
    { required: true, message: '请输入俱乐部名称', trigger: 'blur' },
    { min: 2, max: 50, message: '长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  description: [
    { required: true, message: '请输入俱乐部描述', trigger: 'blur' },
    { min: 1, max: 500, message: '长度在 1 到 500 个字符', trigger: 'blur' }
  ]
}

// 发布活动相关
const activityDialogVisible = ref(false)
const activityFormRef = ref(null)
const publishing = ref(false)
const activityForm = ref({
  title: '',
  content: '',
  time: []
})
const activityRules = {
  title: [
    { required: true, message: '请输入活动标题', trigger: 'blur' },
    { min: 2, max: 50, message: '长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  content: [
    { required: true, message: '请输入活动内容', trigger: 'blur' },
    { min: 1, max: 500, message: '长度在 1 到 500 个字符', trigger: 'blur' }
  ],
  time: [
    { 
      required: true, 
      message: '请选择活动时间', 
      trigger: 'change',
      type: 'array'
    }
  ]
}

// 活动列表相关
const activitiesLoading = ref(false)
const activities = ref([])
const activityPage = ref(1)
const activitySize = ref(2)
const activityTotal = ref(0)

// 判断当前用户是否是创建者
const isCreator = computed(() => {
  const currentUserId = userStore.userInfo?.id;
  const creatorId = clubInfo.value?.creatorId;
  console.log('Creator check details:', {
    currentUserId,
    creatorId,
    currentUserIdType: typeof currentUserId,
    creatorIdType: typeof creatorId,
    userInfo: userStore.userInfo,
    clubInfo: clubInfo.value,
    isMatch: String(creatorId) === String(currentUserId)
  });

  // 确保两个ID都存在
  if (!currentUserId || !creatorId) {
    console.log('Missing userId or creatorId');
    return false;
  }

  // 转换为字符串进行比较
  const isMatch = String(creatorId) === String(currentUserId);
  console.log(`Comparing IDs: ${creatorId} === ${currentUserId} = ${isMatch}`);
  return isMatch;
})

// 判断当前用户是否是成员
const isMember = computed(() => {
  const currentUserId = userStore.userInfo?.id;
  return members.value.some(member => String(member.userId) === String(currentUserId));
})

const loadClubDetails = async () => {
  const clubId = route.params.id
  if (!clubId) {
    ElMessage.error('俱乐部ID不能为空')
    return
  }

  loading.value = true
  try {
    const res = await getClubDetailsService({ clubId })
    console.log('Club details response:', res)
    if (res.success) {
      clubInfo.value = res.data
      console.log('Club info loaded:', clubInfo.value);
      // 加载创建人信息
      if (clubInfo.value.creatorId) {
        await loadCreatorInfo(clubInfo.value.creatorId)
        // 检查创建者状态
        console.log('After loading details:', {
          creatorId: clubInfo.value.creatorId,
          currentUserId: userStore.userInfo?.id,
          isCreator: isCreator.value
        });
      }
    } else {
      ElMessage.error(res.message || '获取俱乐部详情失败')
    }
  } catch (error) {
    console.error('获取俱乐部详情出错:', error)
    ElMessage.error('获取俱乐部详情失败')
  } finally {
    loading.value = false
  }
}

const showRequestsDialog = async () => {
  requestsDialogVisible.value = true
  await loadJoinRequests()
}

const loadCreatorInfo = async (creatorId) => {
  try {
    const res = await getUserInfoByIdService({ id: creatorId })
    console.log('Creator info response:', res)
    if (res.success) {
      creatorInfo.value = res.data
    } else {
      console.warn('获取创建人信息失败:', res.message)
    }
  } catch (error) {
    console.error('获取创建人信息出错:', error)
  }
}

// 加载加入申请列表
const loadJoinRequests = async () => {
  console.log('Starting to load join requests')
  requestsLoading.value = true
  try {
    const params = {
      clubId: route.params.id,
      page: requestPage.value,
      size: requestSize.value
    }
    console.log('Loading join requests with params:', params)
    const res = await getClubJoinRequestsService(params)
    console.log('Join requests response:', res)
    if (res.success) {
      // 获取每个申请的用户信息
      const requests = res.data || []
      const requestsWithUserInfo = await Promise.all(
        requests.map(async (request) => {
          try {
            const userRes = await getUserInfoByIdService({ id: request.userId })
            if (userRes.success) {
              return {
                ...request,
                nickName: userRes.data.nickName,
                avatar: userRes.data.avatar || 'https://cube.elemecdn.com/3/7c/3ea6beec64369c2642b92c6726f1epng.png' // 默认头像
              }
            }
            return request
          } catch (error) {
            console.error('获取用户信息失败:', error)
            return request
          }
        })
      )
      joinRequests.value = requestsWithUserInfo
      console.log('Loaded join requests:', joinRequests.value)
      // 如果返回的数据长度等于页大小，说明可能还有下一页
      requestTotal.value = requestPage.value * requestSize.value + (requests.length === requestSize.value ? requestSize.value : 0)
    } else {
      console.warn('获取申请列表失败:', res.message)
    }
  } catch (error) {
    console.error('获取申请列表出错:', error)
  } finally {
    requestsLoading.value = false
    console.log('Join requests loading finished, current state:', {
      requests: joinRequests.value,
      total: requestTotal.value,
      isCreator: isCreator.value,
      loading: loading.value,
      requestsLoading: requestsLoading.value
    })
  }
}

// 处理申请审核
const handleReview = async (userId, status) => {
  try {
    const res = await reviewJoinRequestService({
      clubId: route.params.id,
      userId,
      status
    })
    if (res.success) {
      // 根据不同的状态显示不同的提示信息
      if (status === 1) {
        ElMessage.success('已同意该用户的加入申请')
      } else if (status === 2) {
        ElMessage({
          type: 'warning',
          message: '已拒绝该用户的加入申请',
          duration: 3000
        })
      }
      loadJoinRequests() // 重新加载申请列表
    } else {
      ElMessage.error(res.message || '处理失败')
    }
  } catch (error) {
    console.error('处理申请出错:', error)
    ElMessage.error(error.response?.data?.message || '处理失败')
  }
}

// 获取状态显示
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

const goToUserDetail = (userId) => {
  router.push(`/user/${userId}`)
}

const goBack = () => {
  router.back()
}

// 显示成员列表对话框
const showMembersDialog = async () => {
  membersDialogVisible.value = true
  await loadMembers()
}

// 加载成员列表
const loadMembers = async () => {
  console.log('Starting to load members');
  membersLoading.value = true
  try {
    const params = {
      clubId: route.params.id,
      page: memberPage.value,
      size: memberSize.value
    };
    console.log('Loading members with params:', params);
    const res = await getClubMembersService(params)
    console.log('Members response:', res);
    if (res.success) {
      members.value = res.data || []
      console.log('Loaded members:', members.value);
      // 使用返回的数组长度作为总数
      memberTotal.value = members.value.length
    } else {
      console.warn('获取成员列表失败:', res.message)
    }
  } catch (error) {
    console.error('获取成员列表出错:', error)
    ElMessage.error('获取成员列表失败')
  } finally {
    membersLoading.value = false
    console.log('Members loading finished, current state:', {
      members: members.value,
      total: memberTotal.value,
      loading: membersLoading.value
    });
  }
}

// 处理退出俱乐部
const handleLeaveClub = async () => {
  try {
    // 显示确认对话框
    await ElMessageBox.confirm(
      '确定要退出该俱乐部吗？',
      '退出确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const res = await leaveClubService({ clubId: route.params.id })
    console.log('Leave club response:', res)
    
    if (res.success) {
      ElMessage.success('退出俱乐部成功')
      // 退出成功后返回上一页
      router.back()
    } else {
      ElMessage.error(res.message || '退出俱乐部失败')
    }
  } catch (error) {
    if (error === 'cancel') {
      return
    }
    console.error('退出俱乐部出错:', error)
    ElMessage.error(error.response?.data?.message || '退出俱乐部失败')
  }
}

// 处理移除成员
const handleRemoveMember = async (userId) => {
  try {
    // 显示确认对话框
    await ElMessageBox.confirm(
      '确定要移除该成员吗？移除后该成员将从俱乐部及相关活动中移除。',
      '移除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const res = await removeMemberService({
      clubId: route.params.id,
      userId
    })
    
    if (res.success) {
      ElMessage.success(res.data || '成员移除成功')
      // 重新加载成员列表
      await loadMembers()
    } else {
      ElMessage.error(res.message || '移除成员失败')
    }
  } catch (error) {
    if (error === 'cancel') {
      return
    }
    console.error('移除成员出错:', error)
    ElMessage.error(error.response?.data?.message || '移除成员失败')
  }
}

// 处理删除俱乐部
const handleDeleteClub = async () => {
  try {
    // 显示确认对话框，使用更严格的警告提示
    await ElMessageBox.confirm(
      '确定要删除该俱乐部吗？删除后将无法恢复，所有成员将被移除，相关活动也将被删除。',
      '删除确认',
      {
        confirmButtonText: '确定删除',
        cancelButtonText: '取消',
        type: 'error',
        confirmButtonClass: 'el-button--danger',
        draggable: true,
        closeOnClickModal: false,
        closeOnPressEscape: false
      }
    )

    const res = await deleteClubService({ clubId: route.params.id })
    console.log('Delete club response:', res)
    
    if (res.success) {
      ElMessage.success('俱乐部删除成功')
      // 删除成功后返回上一页
      router.back()
    } else {
      ElMessage.error(res.message || '删除俱乐部失败')
    }
  } catch (error) {
    if (error === 'cancel') {
      return
    }
    console.error('删除俱乐部出错:', error)
    ElMessage.error(error.response?.data?.message || '删除俱乐部失败')
  }
}

// 显示编辑对话框
const showEditDialog = () => {
  editForm.value = {
    name: clubInfo.value.name,
    description: clubInfo.value.description
  }
  editDialogVisible.value = true
}

// 处理更新俱乐部信息
const handleUpdateClub = async () => {
  if (!editFormRef.value) return
  
  try {
    await editFormRef.value.validate()
    
    updating.value = true
    const res = await updateClubService({
      clubId: route.params.id,
      name: editForm.value.name,
      description: editForm.value.description
    })
    
    if (res.success) {
      ElMessage.success('俱乐部信息更新成功')
      editDialogVisible.value = false
      // 更新成功后重新加载俱乐部详情
      await loadClubDetails()
    } else {
      ElMessage.error(res.message || '更新失败')
    }
  } catch (error) {
    if (error?.message) {
      ElMessage.error(error.message)
    } else {
      console.error('更新俱乐部信息出错:', error)
      ElMessage.error('更新失败')
    }
  } finally {
    updating.value = false
  }
}

// 显示发布活动对话框
const showPublishActivityDialog = () => {
  activityForm.value = {
    title: '',
    content: '',
    time: []
  }
  activityDialogVisible.value = true
}

// 处理发布活动
const handlePublishActivity = async () => {
  if (!activityFormRef.value) return
  
  try {
    await activityFormRef.value.validate()
    
    publishing.value = true
    const [startTime, endTime] = activityForm.value.time
    const res = await publishActivityService({
      clubId: route.params.id,
      title: activityForm.value.title,
      content: activityForm.value.content,
      startTime,
      endTime
    })
    
    if (res.success) {
      ElMessage.success('活动发布成功')
      activityDialogVisible.value = false
      // 重新加载活动列表
      await loadActivities()
    } else {
      ElMessage.error(res.message || '发布失败')
    }
  } catch (error) {
    if (error?.message) {
      ElMessage.error(error.message)
    } else {
      console.error('发布活动出错:', error)
      ElMessage.error('发布失败')
    }
  } finally {
    publishing.value = false
  }
}

// 加载活动列表
const loadActivities = async () => {
  console.log('Starting to load activities');
  activitiesLoading.value = true
  try {
    const params = {
      clubId: route.params.id,
      page: activityPage.value,
      size: activitySize.value
    };
    console.log('Loading activities with params:', params);
    const res = await getClubActivityListService(params)
    console.log('Activities response:', res);
    if (res.success) {
      activities.value = res.data || []
      console.log('Loaded activities:', activities.value);
      // 如果返回的数据长度等于页大小，说明可能还有下一页
      activityTotal.value = activityPage.value * activitySize.value + (activities.value.length === activitySize.value ? activitySize.value : 0)
    } else {
      console.warn('获取活动列表失败:', res.message)
    }
  } catch (error) {
    console.error('获取活动列表出错:', error)
    ElMessage.error('获取活动列表失败')
  } finally {
    activitiesLoading.value = false
  }
}

onMounted(async () => {
  console.log('Component mounted, current user:', userStore.userInfo);
  // 确保用户信息已加载
  if (!userStore.userInfo?.id) {
    console.log('Waiting for user info to load...');
    await userStore.getUserInfo();
    console.log('User info loaded:', userStore.userInfo);
  }
  await loadClubDetails();
  // 加载成员列表以判断当前用户是否是成员
  await loadMembers();
  // 加载活动列表
  await loadActivities();
})
</script>

<style scoped>
.club-detail {
  padding: 32px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 1400px;
  margin: 0 auto;
  min-height: calc(100vh - 64px);
  background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
}

.action-card {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border: none;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  margin-bottom: 24px;
}

.action-card:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.action-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 24px;
  padding: 20px 24px;
}

.club-title {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.club-title h1 {
  margin: 0;
  font-size: 32px;
  font-weight: 700;
  background: linear-gradient(120deg, #1a1a1a, #4a4a4a);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}

.action-buttons {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.action-buttons .el-button {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 10px;
  padding: 12px 24px;
  font-weight: 600;
  letter-spacing: 0.3px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 42px;
}

.action-buttons .el-button:not(.el-button--text) {
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
}

.action-buttons .el-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.content-container {
  display: grid;
  grid-template-columns: minmax(320px, 1fr) 2.2fr;
  gap: 32px;
  position: relative;
}

.left-section {
  display: flex;
  flex-direction: column;
  gap: 32px;
  position: sticky;
  top: 32px;
  height: fit-content;
}

.info-card, .creator-card, .activities-card {
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border: none;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
}

.info-card:hover, .creator-card:hover, .activities-card:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  background: linear-gradient(to right, #f8f9fb, #ffffff);
}

.card-header h3 {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: #1a1a1a;
  position: relative;
  padding-left: 16px;
  letter-spacing: -0.3px;
}

.card-header h3::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 20px;
  background: linear-gradient(to bottom, #409EFF, #36acfe);
  border-radius: 4px;
}

.info-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 24px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: #f8f9fb;
  padding: 20px;
  border-radius: 12px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid rgba(0, 0, 0, 0.04);
}

.info-item:hover {
  background-color: #ecf5ff;
  transform: translateX(8px);
  border-color: rgba(64, 158, 255, 0.2);
}

.info-item label {
  font-weight: 600;
  color: #1a1a1a;
  font-size: 15px;
  opacity: 0.9;
}

.info-item .description {
  line-height: 1.8;
  color: #303133;
  font-size: 15px;
  white-space: pre-wrap;
}

.creator-info-inline {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: 16px;
  padding: 4px 12px;
  background: rgba(0, 0, 0, 0.04);
  border-radius: 20px;
  transition: all 0.3s ease;
}

.creator-info-inline:hover {
  background: rgba(0, 0, 0, 0.08);
}

.creator-label {
  color: #606266;
  font-size: 14px;
}

.creator-name-inline {
  color: #409EFF;
  font-weight: 500;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.creator-name-inline:hover {
  opacity: 0.8;
}

.activity-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 24px;
}

.activity-card {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.activity-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(120deg, rgba(64, 158, 255, 0.1), rgba(54, 172, 254, 0.1));
  opacity: 0;
  transition: opacity 0.3s ease;
}

.activity-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  border-color: rgba(64, 158, 255, 0.3);
}

.activity-card:hover::before {
  opacity: 1;
}

.activity-header {
  margin-bottom: 16px;
}

.activity-header h4 {
  font-size: 18px;
  font-weight: 700;
  color: #1a1a1a;
  margin: 0;
  line-height: 1.4;
}

.activity-meta {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #606266;
  font-size: 14px;
}

.activity-meta .time {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  background: rgba(0, 0, 0, 0.04);
  border-radius: 6px;
}

/* 成员列表优化 */
.members-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 24px;
  padding: 24px;
}

.member-card {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 16px;
}

.member-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  border-color: rgba(64, 158, 255, 0.3);
}

.member-info {
  flex: 1;
}

.member-name {
  font-size: 16px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 8px;
}

.member-join-time {
  font-size: 14px;
  color: #909399;
}

.member-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

/* 申请列表优化 */
.requests-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 24px;
}

.request-item {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 20px;
  border: 1px solid rgba(0, 0, 0, 0.04);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 20px;
}

.request-item:hover {
  transform: translateX(4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  border-color: rgba(64, 158, 255, 0.3);
}

.request-info {
  flex: 1;
}

.request-user {
  font-size: 16px;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 8px;
}

.request-time {
  font-size: 14px;
  color: #909399;
}

.request-actions {
  display: flex;
  gap: 12px;
}

/* 编辑表单优化 */
.edit-form {
  padding: 24px;
}

:deep(.el-form-item) {
  margin-bottom: 24px;
}

:deep(.el-form-item__label) {
  padding-bottom: 8px;
}

:deep(.el-input), :deep(.el-textarea) {
  --el-input-hover-border-color: #409EFF;
  --el-input-focus-border-color: #409EFF;
}

:deep(.el-input__wrapper), :deep(.el-textarea__wrapper) {
  padding: 8px 16px;
}

:deep(.el-input__inner) {
  height: 42px;
}

:deep(.el-textarea__inner) {
  min-height: 120px;
  line-height: 1.8;
}

/* 活动发布表单优化 */
.publish-form {
  padding: 24px;
}

.time-picker-wrapper {
  display: flex;
  gap: 16px;
}

:deep(.el-date-editor) {
  width: 100%;
}

:deep(.el-date-editor .el-range-separator) {
  padding: 0 12px;
}
</style> 