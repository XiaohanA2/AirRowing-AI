<template>
  <div class="my-notes-container">
    <el-card class="note-detail-card" shadow="hover">
      <div class="social-stats">
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
      </div>

      <div class="header-actions">
        <div class="tab-buttons">
          <button 
            class="tab-btn" 
            :class="{ active: currentTab === 'notes' }"
            @click="switchTab('notes')"
          >
            我的帖子
          </button>
          <button 
            class="tab-btn" 
            :class="{ active: currentTab === 'likes' }"
            @click="switchTab('likes')"
          >
            我的点赞
          </button>
        </div>
        
        <button 
          class="publish-btn" 
          @click="openPublishModal"
          v-if="status === 1 && currentTab === 'notes'"
        >
          发布帖子
        </button>
      </div>

      <div v-if="loading" class="loading-spinner">
        <div class="spinner"></div>
      </div>

      <div v-else>
        <div v-if="error" class="error-message">
          {{ error }}
        </div>

        <div v-else>
          <div class="notes-grid">
            <template v-if="currentTab === 'notes'">
              <div 
                class="my-note-card" 
                :class="{ 'is-top': note.isTop }" 
                v-for="note in paginatedNotes" 
                :key="note.id"
                @click="goToNoteDetail(note.id)"
              >
                <div class="note-images">
                  <img src="/assets/developer.png" alt="默认图片" class="note-image" />
                  <img v-if="note.imgUris && note.imgUris.length > 0" 
                       :src="note.imgUris[0]" 
                       alt="" 
                       class="note-image" 
                       style="position: absolute; top: 0; left: 0;" />
                </div>
                <h3>{{ note.title }}</h3>
                <small>{{ formatDate(note.updateTime) }}</small>
                <div class="note-actions">
                  <button class="edit-btn" @click.stop="editNote(note)" v-if="status === 1">编辑</button>
                  <button class="top-btn" @click.stop="toggleTop(note)" :class="{ active: note.isTop }">
                    {{ note.isTop ? '取消置顶' : '置顶' }}
                  </button>
                  <button class="visibility-btn" @click.stop="toggleVisibility(note)" :class="{ active: note.visible === 1 }">
                    {{ note.visible === 1 ? '公开' : '仅自己可见' }}
                  </button>
                  <button class="delete-btn" v-if="status === 1" @click.stop="deleteNote(note)">
                    删除
                  </button>
                  <button class="restore-btn" v-else @click.stop="restoreNote(note)">
                    恢复
                  </button>
                </div>
              </div>
            </template>

            <template v-else>
              <div 
                class="my-note-card"
                v-for="article in likedNotes"
                :key="article.id"
                @click="goToNoteDetail(article.id)"
              >
                <div class="note-images">
                  <img src="/assets/developer.png" alt="默认图片" class="note-image" />
                  <img 
                    v-if="article.imgUris" 
                    :src="article.imgUris" 
                    alt="" 
                    class="note-image"
                    style="position: absolute; top: 0; left: 0;"
                  />
                </div>
                <h3>{{ article.title }}</h3>
                <div class="note-info">
                  <div class="creator-info">
                    <img :src="article.avatar || '/assets/developer.png'" alt="Creator Avatar" class="creator-avatar" />
                    <span class="creator-name">{{ article.creatorName }}</span>
                  </div>
                  <small class="update-time">{{ formatDate(article.updateTime) }}</small>
                </div>
              </div>
            </template>
          </div>

          <div class="pagination-container">
            <el-pagination
              v-model:current-page="currentPage"
              :page-size="pageSize"
              :total="currentTab === 'notes' ? myNotes.length : likedNotes.length"
              layout="prev, pager, next"
              @current-change="handlePageChange"
            />
          </div>
        </div>
      </div>

      <transition name="fade">
        <div v-if="isEditModalOpen" class="modal-overlay" @click.self="closeEditModal">
          <div class="modal">
            <h2>编辑帖子</h2>
            <form @submit.prevent="submitEditNote" class="form">
              <label>
                标题:
                <input type="text" v-model="currentNote.title" required />
              </label>
              <label>
                内容:
                <textarea v-model="currentNote.content" required></textarea>
              </label>
              <label>
                上传图片:
                <input type="file" multiple @change="handleEditImageUpload" accept="image/*" />
              </label>
              <div class="image-previews">
                <div v-for="(img, index) in currentNote.imgUris" :key="index" class="image-preview">
                  <img :src="img" alt="预览图片" />
                  <button type="button" @click="removeImage(currentNote.imgUris, index)">×</button>
                </div>
              </div>
              <div class="form-buttons">
                <button type="submit" class="btn save-btn" :disabled="loading">保存修改</button>
                <button type="button" class="btn cancel-btn" @click="closeEditModal">取消</button>
              </div>
            </form>
          </div>
        </div>
      </transition>

      <transition name="fade">
        <div v-if="isPublishModalOpen" class="modal-overlay" @click.self="closePublishModal">
          <div class="modal">
            <h2>发布帖子</h2>
            <form @submit.prevent="submitPublishNote" class="form">
              <label>
                标题:
                <input type="text" v-model="newNote.title" required />
              </label>
              <label>
                内容:
                <textarea v-model="newNote.content" required></textarea>
              </label>
              <label>
                上传图片 (最多4张):
                <input 
                  type="file" 
                  multiple 
                  @change="handlePublishImageUpload" 
                  accept="image/*"
                  :disabled="newNote.imgUris.length >= 4" 
                />
              </label>
              <div class="image-previews">
                <div v-for="(img, index) in newNote.imgUris" :key="index" class="image-preview">
                  <img :src="img" alt="预览图片" />
                  <button type="button" @click="removeImage(newNote.imgUris, index)">×</button>
                </div>
              </div>
              <div class="form-buttons">
                <button type="submit" class="btn save-btn" :disabled="loading">发布</button>
                <button type="button" class="btn cancel-btn" @click="closePublishModal">取消</button>
              </div>
            </form>
          </div>
        </div>
      </transition>

      <div v-if="!loading && currentTab === 'notes' && myNotes.length === 0" class="empty-state">
        <img src="/assets/developer.png" alt="没有帖子" />
        <h3>还没有帖子</h3>
        <p>点击"发布帖子"开始创作吧！</p>
      </div>

      <transition name="fade">
        <div v-if="showMessage" class="message-toast" :class="messageType">
          {{ message }}
        </div>
      </transition>

      <el-dialog
        v-model="deleteDialogVisible"
        title="删除确认"
        width="30%"
        center
        :show-close="false"
        class="delete-dialog"
      >
        <div class="delete-dialog-content">
          <i class="el-icon-warning" style="color: #ff4949; font-size: 24px;"></i>
          <p>确定要删除这篇帖子吗？</p>
          <p class="delete-warning">删除后将无法恢复！</p>
        </div>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="deleteDialogVisible = false">取消</el-button>
            <el-button type="danger" @click="confirmDelete" :loading="loading">
              确定删除
            </el-button>
          </span>
        </template>
      </el-dialog>

      <el-drawer
        v-model="socialDrawerVisible"
        :title="drawerType === 'following' ? '关注列表' : '粉丝列表'"
        direction="rtl"
        size="35%"
      >
        <div class="user-list">
          <el-table
            :data="userList"
            style="width: 100%"
            v-loading="loading"
          >
            <el-table-column label="用户信息" min-width="200">
              <template #default="{ row }">
                <div class="user-info">
                  <el-avatar :size="40" :src="row.avatar" />
                  <div class="user-details">
                    <div class="nickname">{{ row.nickname }}</div>
                    <div class="introduction">{{ row.introduction || '这个人很懒，什么都没写~' }}</div>
                  </div>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="120" align="center">
              <template #default="{ row }">
                <el-button
                  type="primary"
                  size="small"
                  @click="goToUserDetail(row.userId)"
                >
                  查看主页
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-drawer>
    </el-card>
  </div>
</template>

<script>
import axios from 'axios';
import { getToken } from '@/composables/cookie';
import { ref, reactive, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { likeListService, collectListDetailService } from '@/api/collect.js'
import { UserFilled, Star } from '@element-plus/icons-vue';
import { currentUserService, followService, fansService } from '@/api/follow.js';

export default {
  setup() {
    const router = useRouter();
    const myNotes = ref([]);
    const userId = ref(null);
    const size = ref(10);
    const page = ref(1);
    const loading = ref(false);
    const error = ref(null);
    const noMoreNotes = ref(false);
    const currentTab = ref('notes');
    const likedNotes = ref([]);

    const isEditModalOpen = ref(false);
    const isPublishModalOpen = ref(false);
    const currentNote = reactive({
      id: null,
      title: "",
      content: "",
      updateTime: "",
      isTop: false,
      visible: 1,
      imgUris: []
    });
    const newNote = reactive({
      type: 0,
      imgUris: [],
      title: "",
      content: "",
      topicId: 1
    });

    const status = ref(1);

    const currentPage = ref(1);
    const pageSize = ref(9);

    const paginatedNotes = computed(() => {
      const startIndex = (currentPage.value - 1) * pageSize.value;
      const endIndex = startIndex + pageSize.value;
      return sortedNotes.value.slice(startIndex, endIndex);
    });

    const handlePageChange = (newPage) => {
      currentPage.value = newPage;
      if (newPage * pageSize.value > myNotes.value.length - pageSize.value && !noMoreNotes.value) {
        loadMoreNotes();
      }
    };

    const formatDate = (dateString) => {
      if (!dateString) return '';

      const [datePart, timePart] = dateString.split(' ');
      if (!datePart || !timePart) return '';

      const [year, month, day] = datePart.split('-');
      const [hour, minute, second] = timePart.split(':');

      const date = new Date(year, month - 1, day, hour, minute, second);

      if (isNaN(date.getTime())) return '';

      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
    };

    const fetchNoteDetails = async (note, token) => {
      try {
        const response = await axios.post(
          '/api/note/note/detail',
          { id: note.id },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (response.data.success) {
          note.imgUris = response.data.data.imgUris || [];
          note.updateTime = response.data.data.updateTime;
        } else {
          console.error(`获取帖子详情失败（ID: ${note.id}）:`, response.data.message);
          note.imgUris = [];
        }
      } catch (err) {
        console.error(`获取帖子详情时出错（ID: ${note.id}）:`, err);
        note.imgUris = [];
      }
    };

    const fetchCurrentUser = async () => {
      try {
        const token = getToken();
        if (!token) {
          throw new Error('未找到认证信息，请重新登录。');
        }
        const response = await axios.get('/api/user/user/current', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (response.data.success) {
          userId.value = response.data.data.id;
          fetchMyNotes();
        } else {
          throw new Error(response.data.message || '获取用户信息失败。');
        }
      } catch (err) {
        console.error('获取用户信息时出错:', err);
        error.value = err.message || '获取用户信息时出错。';
      }
    };

    const fetchMyNotes = async () => {
      if (!userId.value) {
        error.value = '无法获取用户 ID，无法获取帖子列表。';
        return;
      }
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/UserNoteList',
          {
            userId: userId.value,
            size: size.value * 2,
            page: page.value,
            status: status.value
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          const fetchedNotes = response.data.data.map(note => ({
            ...note,
            imgUris: note.imgUris || [],
            isTop: !!note.isTop
          }));
          if (fetchedNotes.length < size.value) {
            noMoreNotes.value = true;
          }

          const detailPromises = fetchedNotes.map(note => fetchNoteDetails(note, token));
          await Promise.all(detailPromises);

          myNotes.value = [...myNotes.value, ...fetchedNotes];
        } else {
          throw new Error(response.data.message || '获取帖子列表失败。');
        }
      } catch (err) {
        console.error('获取帖子列表时出错:', err);
        error.value = err.message || '获取帖子列表时出错。';
      } finally {
        loading.value = false;
      }
    };

    const sortedNotes = computed(() => {
      return myNotes.value.slice().sort((a, b) => {
        if (a.isTop === b.isTop) {
          return new Date(b.updateTime) - new Date(a.updateTime);
        }
        return b.isTop - a.isTop;
      });
    });

    const changeStatus = (newStatus) => {
      if (status.value !== newStatus) {
        status.value = newStatus;
        myNotes.value = [];
        page.value = 1;
        noMoreNotes.value = false;
        fetchMyNotes();
      }
    };

    const loadMoreNotes = () => {
      if (!noMoreNotes.value && !loading.value) {
        page.value++;
        fetchMyNotes();
      }
    };

    const editNote = (note) => {
      Object.assign(currentNote, note);
      currentNote.imgUris = note.imgUris ? [...note.imgUris] : [];
      isEditModalOpen.value = true;
    };

    const submitEditNote = async () => {
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/update',
          {
            id: currentNote.id,
            type: 0,
            title: currentNote.title,
            content: currentNote.content,
            imgUris: currentNote.imgUris,
            topicId: 1,
            videoUri: null
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          const index = myNotes.value.findIndex(note => note.id === currentNote.id);
          if (index !== -1) {
            myNotes.value[index] = {
              ...myNotes.value[index],
              title: currentNote.title,
              content: currentNote.content,
              imgUris: [...currentNote.imgUris],
              updateTime: new Date().toISOString(),
              isTop: currentNote.isTop
            };
          }
          showToast('帖子已更新', 'success');
          closeEditModal();
        } else {
          throw new Error(response.data.message || '更新帖子失败。');
        }
      } catch (err) {
        console.error('更新帖子时出错:', err);
        showToast(err.message || '更新帖子时出错。', 'error');
      } finally {
        loading.value = false;
      }
    };

    const closeEditModal = () => {
      isEditModalOpen.value = false;
      Object.assign(currentNote, {
        id: null,
        title: "",
        content: "",
        updateTime: "",
        isTop: false,
        visible: 1,
        imgUris: []
      });
    };

    const openPublishModal = () => {
      isPublishModalOpen.value = true;
    };

    const closePublishModal = () => {
      isPublishModalOpen.value = false;
      Object.assign(newNote, {
        type: 0,
        imgUris: [],
        title: "",
        content: "",
        topicId: 1
      });
    };

    const submitPublishNote = async () => {
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/publish',
          {
            type: newNote.type,
            imgUris: newNote.imgUris,
            title: newNote.title,
            content: newNote.content,
            topicId: newNote.topicId
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          alert('帖子已发布。');
          console.log(newNote.imgUris);
          myNotes.value = [];
          page.value = 1;
          noMoreNotes.value = false;
          fetchMyNotes();
          closePublishModal();
        } else {
          throw new Error(response.data.message || '发布帖子失败。');
        }
      } catch (err) {
        console.error('发布帖子时出错:', err);
        alert(err.message || '发布帖子时出错。');
      } finally {
        loading.value = false;
      }
    };

    const handlePublishImageUpload = async (event) => {
      const files = event.target.files;
      if (!files.length) return;

      if (newNote.imgUris.length + files.length > 4) {
        alert('最多只能上传4张图片');
        event.target.value = '';
        return;
      }

      const uploadedImageUrls = [];

      for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
          const response = await axios.post('/file/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
              'Authorization': `Bearer ${getToken()}`,
            },
          });

          if (response.data.success) {
            uploadedImageUrls.push(response.data.data);
          } else {
            throw new Error(response.data.message || '图片上传失败');
          }
        } catch (err) {
          console.error('上传图片失败:', err);
          alert('上传图片失败: ' + (err.response?.data?.message || err.message));
          return;
        }
      }

      newNote.imgUris = [...newNote.imgUris, ...uploadedImageUrls];
      event.target.value = '';
    };

    const handleEditImageUpload = async (event) => {
      const files = event.target.files;
      if (!files.length) return;

      const uploadedImageUrls = [];

      for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
          const response = await axios.post('/file/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
              'Authorization': `Bearer ${getToken()}`,
            },
          });

          if (response.data.success) {
            uploadedImageUrls.push(response.data.data);
          } else {
            throw new Error(response.data.message || '图片上传失败');
          }
        } catch (err) {
          console.error('上传图片失败:', err);
          alert('上传图片失败: ' + (err.response?.data?.message || err.message));
          return;
        }
      }

      currentNote.imgUris = [...currentNote.imgUris, ...uploadedImageUrls];

      event.target.value = '';
    };

    const removeImage = (imgUrisArray, index) => {
      imgUrisArray.splice(index, 1);
    };

    const deleteDialogVisible = ref(false);
    const noteToDelete = ref(null);

    const deleteNote = (note) => {
      noteToDelete.value = note;
      deleteDialogVisible.value = true;
    };

    const confirmDelete = async () => {
      if (!noteToDelete.value) return;
      
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/delete',
          {
            type: noteToDelete.value.type,
            videoUri: noteToDelete.value.videoUri || null,
            title: noteToDelete.value.title,
            content: noteToDelete.value.content,
            topicId: noteToDelete.value.topicId,
            id: noteToDelete.value.id
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        
        if (response.data.success) {
          showToast('帖子删除成功', 'success');
          myNotes.value = myNotes.value.filter(n => n.id !== noteToDelete.value.id);
          deleteDialogVisible.value = false;
        } else {
          throw new Error(response.data.message || '删除帖子失败。');
        }
      } catch (err) {
        console.error('删除帖子时出错:', err);
        showToast(err.message || '删除帖子失败', 'error');
      } finally {
        loading.value = false;
        noteToDelete.value = null;
      }
    };

    const restoreNote = async (note) => {
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/restore',
          {
            id: note.id
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          alert('帖子已恢复。');
          myNotes.value = myNotes.value.filter(n => n.id !== note.id);
        } else {
          throw new Error(response.data.message || '恢复帖子失败。');
        }
      } catch (err) {
        console.error('恢复帖子时出错:', err);
        alert(err.message || '恢复帖子时出错。');
      } finally {
        loading.value = false;
      }
    };

    const toggleTop = async (note) => {
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/top',
          {
            id: note.id,
            isTop: !note.isTop
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          note.isTop = !note.isTop;
          alert(note.isTop ? '帖子已置顶。' : '帖子已取消置顶。');
        } else {
          throw new Error(response.data.message || '操作失败。');
        }
      } catch (err) {
        console.error('置顶帖子时出错:', err);
        alert(err.message || '操作时出错。');
      } finally {
        loading.value = false;
      }
    };

    const toggleVisibility = async (note) => {
      try {
        loading.value = true;
        const token = getToken();
        const response = await axios.post(
          '/api/note/note/visible/onlyme',
          {
            id: note.id,
            visible: note.visible === 1 ? 0 : 1
          },
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.data.success) {
          note.visible = note.visible === 1 ? 0 : 1;
          alert(note.visible === 1 ? '帖子现在公开可见。' : '帖子现在仅自己可见。');
        } else {
          throw new Error(response.data.message || '操作失败。');
        }
      } catch (err) {
        console.error('切换可见性时出错:', err);
        alert(err.message || '操作时出错。');
      } finally {
        loading.value = false;
      }
    };

    const fetchLikedNotes = async () => {
      try {
        loading.value = true;
        const params = {
          size: 1000,
          userId: userId.value,
          page: 1
        };
        const result = await likeListService(params);
        
        const updatedArticles = await Promise.all(
          result.data.map(async (article) => {
            const detailResult = await collectListDetailService({ id: article.id });
            return {
              ...article,
              ...detailResult.data,
              imgUris: detailResult.data.imgUris,
              creatorName: detailResult.data.creatorName || '未知用户',
              avatar: detailResult.data.avatar || '/assets/developer.png',
              updateTime: detailResult.data.updateTime
            };
          })
        );
        
        likedNotes.value = updatedArticles;
      } catch (err) {
        console.error('获取点赞列表失败:', err);
        error.value = '获取点赞列表失败';
      } finally {
        loading.value = false;
      }
    };

    const switchTab = async (tab) => {
      currentTab.value = tab;
      currentPage.value = 1;
      
      if (tab === 'likes' && likedNotes.value.length === 0) {
        await fetchLikedNotes();
      }
    };

    const goToNoteDetail = (id) => {
      router.push({
        name: 'NoteDetail',
        params: {
          id,
          userId: userId.value
        }
      });
    };

    const showMessage = ref(false);
    const message = ref('');
    const messageType = ref('success');

    const showToast = (msg, type = 'success') => {
      message.value = msg;
      messageType.value = type;
      showMessage.value = true;
      setTimeout(() => {
        showMessage.value = false;
      }, 3000);
    };

    const socialDrawerVisible = ref(false);
    const drawerType = ref('following');
    const followCount = ref(0);
    const fansCount = ref(0);
    const userList = ref([]);

    const showFollowings = async () => {
      drawerType.value = 'following';
      userList.value = [];
      socialDrawerVisible.value = true;
      await loadUsers();
    };

    const showFans = async () => {
      drawerType.value = 'fans';
      userList.value = [];
      socialDrawerVisible.value = true;
      await loadUsers();
    };

    const loadUsers = async () => {
      if (loading.value) return;
      loading.value = true;
      
      try {
        const params = {
          userId: userId.value,
          pageNo: 1,
          pageSize: 20
        };
        
        const service = drawerType.value === 'following' ? followService : fansService;
        const result = await service(params);
        
        if (result.success) {
          if (result.data && Array.isArray(result.data)) {
            userList.value = result.data.map(user => ({
              ...user,
              avatar: user.avatar || '/assets/developer.png',
              nickname: user.nickname || '未设置昵称',
              introduction: user.introduction || '这个人很懒，什么都没写~'
            }));
          } else {
            userList.value = [];
          }
        } else {
          throw new Error(result.message || '获取用户列表失败');
        }
      } catch (error) {
        console.error('加载用户失败:', error);
        ElMessage.error(error.message || '加载用户列表失败');
      } finally {
        loading.value = false;
      }
    };

    const goToUserDetail = (userId) => {
      router.push(`/user/${userId}`);
      socialDrawerVisible.value = false;
    };

    const initSocialData = async () => {
      try {
        const result = await currentUserService();
        if (result.success) {
          userId.value = result.data.id;
          const [followRes, fansRes] = await Promise.all([
            followService({ 
              userId: userId.value, 
              pageNo: 1,
              pageSize: 1
            }),
            fansService({ 
              userId: userId.value, 
              pageNo: 1,
              pageSize: 1
            })
          ]);

          if (followRes.success) {
            followCount.value = followRes.totalCount || 0;
          }
          if (fansRes.success) {
            fansCount.value = fansRes.totalCount || 0;
          }
        }
      } catch (error) {
        console.error('初始化社交数据失败:', error);
      }
    };

    onMounted(() => {
      initSocialData();
    });

    fetchCurrentUser();

    return {
      myNotes,
      sortedNotes,
      status,
      loading,
      error,
      noMoreNotes,
      loadMoreNotes,
      changeStatus,
      editNote,
      isEditModalOpen,
      currentNote,
      submitEditNote,
      closeEditModal,
      formatDate,
      isPublishModalOpen,
      openPublishModal,
      closePublishModal,
      newNote,
      submitPublishNote,
      deleteNote,
      restoreNote,
      toggleTop,
      toggleVisibility,
      handlePublishImageUpload,
      handleEditImageUpload,
      removeImage,
      goToNoteDetail,
      currentPage,
      pageSize,
      paginatedNotes,
      handlePageChange,
      showMessage,
      message,
      messageType,
      showToast,
      currentTab,
      likedNotes,
      switchTab,
      deleteDialogVisible,
      noteToDelete,
      confirmDelete,
      socialDrawerVisible,
      drawerType,
      followCount,
      fansCount,
      userList,
      showFollowings,
      showFans,
      goToUserDetail,
    };
  },
};
</script>

<style scoped>
.my-notes-container {
  display: flex;
  justify-content: center;
  width: 100%;
  min-height: calc(100vh - 120px);
  background-color: #f5f5f5;
  padding: 20px;
  box-sizing: border-box;
}

.note-detail-card {
  width: 90%;
  max-width: 1200px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  height: fit-content;
  min-height: calc(100vh - 160px);
  position: relative;
}

h1 {
  text-align: center;
  margin-bottom: 30px;
  color: #343a40;
}

.filter-actions {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
  gap: 10px;
}

.filter-btn {
  padding: 8px 16px;
  font-size: 1em;
  color: #fff;
  background-color: #6c757d;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.filter-btn.active {
  background-color: #007bff;
}

.filter-btn:hover:not(.active) {
  background-color: #5a6268;
}

.header-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #ebeef5;
}

.publish-btn {
  padding: 8px 24px;
  font-size: 14px;
  color: #fff;
  background-color: #409eff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.publish-btn:hover {
  background-color: #66b1ff;
}

.notes-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  padding: 20px;
  margin-bottom: 60px;
}

.my-note-card {
  background: #ffffff;
  padding: 20px;
  border-radius: 15px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  position: relative;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
  cursor: pointer;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.my-note-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
  border-color: rgba(0, 123, 255, 0.3);
}

.my-note-card.is-top {
  border-left: 5px solid #28a745;
}

.my-note-card h3 {
  font-size: 1.5em;
  margin: 15px 0 10px 0;
  color: #007bff;
}

.my-note-card p {
  font-size: 1em;
  color: #555;
  margin-bottom: 15px;
  flex-grow: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

.my-note-card small {
  font-size: 0.8em;
  color: #999;
}

.note-images {
  width: 100%;
  height: 200px;
  position: relative;
  overflow: hidden;
  border-radius: 10px;
  background: #f8f9fa;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 15px;
}

.note-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.note-images img[src="/assets/developer.png"] {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.note-images:hover .note-image {
  transform: scale(1.05);
}

.note-actions {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 15px;
}

.edit-btn,
.delete-btn,
.top-btn,
.visibility-btn,
.restore-btn {
  padding: 6px 12px;
  font-size: 0.9em;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 4px;
  font-weight: 500;
}

.edit-btn {
  background-color: #ffc107;
  color: #212529;
}

.edit-btn:hover {
  background-color: #e0a800;
}

.delete-btn {
  background-color: #dc3545;
  color: #fff;
}

.delete-btn:hover {
  background-color: #c82333;
}

.restore-btn {
  background-color: #17a2b8;
  color: #fff;
}

.restore-btn:hover {
  background-color: #138496;
}

.top-btn {
  background-color: #28a745;
  color: #fff;
}

.top-btn.active {
  background-color: #218838;
}

.top-btn:hover:not(.active) {
  background-color: #218838;
}

.visibility-btn {
  background-color: #17a2b8;
  color: #fff;
}

.visibility-btn.active {
  background-color: #138496;
}

.visibility-btn:hover:not(.active) {
  background-color: #138496;
}

.load-more-container {
  display: flex;
  justify-content: center;
  margin-top: 30px;
}

.load-more {
  padding: 10px 20px;
  font-size: 1em;
  color: #fff;
  background-color: #28a745;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.load-more:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
}

.load-more:hover:not(:disabled) {
  background-color: #218838;
}

.error-message {
  color: red;
  text-align: center;
  margin-top: 20px;
  font-weight: bold;
}

.loading-spinner {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
}

.loading-spinner::after {
  content: "";
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal {
  background: #ffffff;
  padding: 30px;
  border-radius: 20px;
  width: 90%;
  max-width: 600px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  animation: modalFadeIn 0.3s ease;
}

@keyframes modalFadeIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal h2 {
  margin-bottom: 20px;
  color: #343a40;
}

.form {
  display: flex;
  flex-direction: column;
}

.form label {
  margin-bottom: 15px;
  color: #333;
  font-weight: bold;
}

.form input,
.form textarea {
  width: 100%;
  padding: 12px;
  margin-top: 8px;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  box-sizing: border-box;
  transition: border-color 0.3s ease;
}

.form input:focus,
.form textarea:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.image-previews {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
}

.image-preview {
  position: relative;
  width: 100px;
  height: 100px;
}

.image-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 5px;
}

.image-preview button {
  position: absolute;
  top: 5px;
  right: 5px;
  background-color: rgba(220, 53, 69, 0.8);
  border: none;
  color: #fff;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  cursor: pointer;
  font-size: 0.8em;
  line-height: 20px;
  text-align: center;
  padding: 0;
}

.image-preview button:hover {
  background-color: rgba(220, 53, 69, 1);
}

.form-buttons {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 20px;
}

.save-btn {
  background-color: #28a745;
  color: #fff;
  border: none;
  padding: 8px 16px;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.save-btn:hover {
  background-color: #218838;
}

.cancel-btn {
  background-color: #6c757d;
  color: #fff;
  border: none;
  padding: 8px 16px;
  border-radius: 5px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.cancel-btn:hover {
  background-color: #5a6268;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .notes-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .note-images img[src="/assets/developer.png"] {
    width: 70%;
  }
}

@media (max-width: 480px) {
  .my-notes-container {
    padding: 10px;
  }

  .notes-grid {
    grid-template-columns: 1fr;
    padding: 15px;
  }
  
  .note-images img[src="/assets/developer.png"] {
    width: 80%;
  }
  
  .pagination-container {
    padding: 8px 15px;
    bottom: 10px;
  }
}

.pagination-container {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  width: fit-content;
  background: #fff;
  padding: 10px 20px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  z-index: 100;
}

:deep(.el-pagination) {
  text-align: center;
  font-size: 14px;
  padding: 10px 20px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

:deep(.el-pagination .el-pager li) {
  min-width: 32px;
  height: 32px;
  line-height: 32px;
  border-radius: 4px;
  margin: 0 3px;
  font-weight: 500;
}

:deep(.el-pagination .el-pager li.active) {
  background-color: #409eff;
  color: #fff;
}

:deep(.el-pagination .btn-prev),
:deep(.el-pagination .btn-next) {
  min-width: 32px;
  height: 32px;
  border-radius: 4px;
  padding: 0;
  margin: 0 3px;
}

@media (max-width: 480px) {
  :deep(.el-pagination) {
    font-size: 12px;
    padding: 8px 12px;
  }

  :deep(.el-pagination .el-pager li) {
    min-width: 28px;
    height: 28px;
    line-height: 28px;
    margin: 0 2px;
  }

  :deep(.el-pagination .btn-prev),
  :deep(.el-pagination .btn-next) {
    min-width: 28px;
    height: 28px;
  }
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  text-align: center;
  min-height: 400px;
}

.empty-state img {
  width: 160px;
  height: 160px;
  object-fit: contain;
  margin-bottom: 24px;
  opacity: 0.6;
}

.empty-state h3 {
  color: #606266;
  font-size: 18px;
  margin-bottom: 12px;
  font-weight: 500;
}

.empty-state p {
  color: #909399;
  font-size: 14px;
  margin: 0;
}

.message-toast {
  position: fixed;
  top: 20px;
  right: 20px;
  padding: 10px 20px;
  background-color: #fff;
  border-radius: 5px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  color: #333;
  font-size: 1em;
  font-weight: bold;
  animation: toastFadeIn 0.3s ease;
}

@keyframes toastFadeIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-toast.success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
}

.message-toast.error {
  background-color: #f2dede;
  border-color: #ebccd1;
}

input[type="file"]:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.image-upload-info {
  font-size: 0.8em;
  color: #666;
  margin-top: 4px;
}

.tab-buttons {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
  padding: 0 20px;
}

.tab-btn {
  padding: 8px 24px;
  font-size: 14px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: #f5f7fa;
  color: #606266;
}

.tab-btn.active {
  background-color: #409eff;
  color: #fff;
}

.tab-btn:hover:not(.active) {
  background-color: #e4e7ed;
}

@media (max-width: 480px) {
  .tab-buttons {
    padding: 0 15px;
  }
  
  .tab-btn {
    padding: 6px 16px;
    font-size: 13px;
  }
}

.note-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
}

.creator-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.creator-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid #ebeef5;
}

.creator-name {
  font-size: 13px;
  color: #606266;
}

.update-time {
  font-size: 12px;
  color: #909399;
}

@media (max-width: 480px) {
  .header-actions {
    padding: 15px;
  }
  
  .publish-btn {
    padding: 6px 16px;
    font-size: 13px;
  }
}

.delete-dialog :deep(.el-dialog__header) {
  border-bottom: 1px solid #ebeef5;
  padding: 20px;
  margin: 0;
}

.delete-dialog :deep(.el-dialog__body) {
  padding: 30px 20px;
}

.delete-dialog :deep(.el-dialog__footer) {
  border-top: 1px solid #ebeef5;
  padding: 15px 20px;
}

.delete-dialog-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 15px;
  text-align: center;
}

.delete-warning {
  color: #ff4949;
  font-size: 14px;
  margin-top: 5px;
}

.dialog-footer {
  display: flex;
  justify-content: center;
  gap: 15px;
}

.message-toast {
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2000;
  padding: 12px 24px;
  border-radius: 4px;
  font-size: 14px;
  transition: all 0.3s;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  display: flex;
  align-items: center;
  gap: 8px;
}

.message-toast.success {
  background-color: #f0f9eb;
  border: 1px solid #e1f3d8;
  color: #67c23a;
}

.message-toast.error {
  background-color: #fef0f0;
  border: 1px solid #fde2e2;
  color: #f56c6c;
}

.social-stats {
  margin-bottom: 20px;
  padding: 0 20px;
}

.stat-cards {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.stat-card {
  cursor: pointer;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-content {
  display: flex;
  align-items: center;
  padding: 16px;
}

.stat-icon {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  margin-right: 16px;
  font-size: 20px;
}

.following-icon {
  background-color: #ecf5ff;
  color: #409eff;
}

.fans-icon {
  background-color: #fdf6ec;
  color: #e6a23c;
}

.stat-info {
  flex: 1;
}

.stat-number {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}

.stat-label {
  color: #909399;
  font-size: 14px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-details {
  display: flex;
  flex-direction: column;
}

.nickname {
  font-weight: 500;
  color: #303133;
}

.introduction {
  font-size: 12px;
  color: #909399;
}

@media (max-width: 768px) {
  .stat-cards {
    grid-template-columns: 1fr;
  }
}
</style>
