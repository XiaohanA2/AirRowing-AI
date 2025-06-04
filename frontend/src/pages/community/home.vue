<template>
  <div class="page-container">
    <el-card class="content-card" shadow="never">
      <div class="header-section">
        <div class="header-content">
          <div class="title-section">
            <h2 class="text-2xl font-bold">社区动态</h2>
            <p class="text-gray-600 mt-2">发现精彩的赛艇故事</p>
          </div>
          
          <div class="search-bar">
            <el-input
              v-model="searchQuery"
              placeholder="搜索帖子..."
              :prefix-icon="Search"
              clearable
              @input="handleSearch"
              class="search-input"
            >
              <template #append>
                <el-button :icon="Search" @click="handleSearch">
                  搜索
                </el-button>
              </template>
            </el-input>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <template v-if="filteredNotes.length > 0">
          <el-card v-for="note in filteredNotes.filter(note => !note.placeholder)" 
                  :key="note.id"
                  class="note-card"
                  shadow="hover"
                  @click="goToNoteDetail(note.id, note.creatorId)">
            <div class="image-container mb-4">
              <div class="image-wrapper">
                <img 
                  :src="note.imgUris && note.imgUris[0] ? note.imgUris[0] : '/assets/developer.png'"
                  @error="handleImageError" 
                  alt="Note Image" 
                  class="note-image"
                  :style="getImageStyle(note)"
                />
                <div class="image-overlay"></div>
              </div>
            </div>

            <h3 class="text-lg font-semibold mb-2 line-clamp-2">{{ note.title }}</h3>
            
            <div class="flex items-center justify-between mt-4">
              <div class="creator-info flex items-center">
                <img :src="note.avatar" alt="Creator Avatar" class="creator-avatar" />
                <span class="ml-2 text-sm text-gray-600">{{ note.creatorName }}</span>
              </div>
              <span class="text-sm text-gray-500">{{ note.updateTime }}</span>
            </div>
          </el-card>
        </template>
      </div>

      <div v-if="searchQuery && filteredNotesWithoutPlaceholders.length === 0" class="empty-search-result">
        <img src="/assets/developer.png" alt="无搜索结果" class="empty-image" />
        <p class="empty-text">没有找到相关帖子</p>
      </div>

      <div class="flex justify-center mt-8">
        <el-pagination
          v-model:current-page="currentPage"
          :page-size="pageSize"
          :total="totalNotes"
          :page-count="maxPage"
          layout="prev, jumper, next"
          @current-change="handlePageChange"
          background
        />
      </div>
    </el-card>
  </div>
</template>

<script>
import { getNoteListService, getNoteDetailService } from "@/api/note.js";
import { ElMessage } from "element-plus";
import { Search } from '@element-plus/icons-vue'

export default {
  data() {
    return {
      notes: [], // 当前页的笔记数据
      currentPage: 1, // 当前页码
      currentPageInput: 1, // 输入框中的页码
      pageSize: 9, // 每页固定展示 9 条数据
      totalNotes: 0, // 总笔记数，从后端获取
      imageStyles: {}, // 存储每个图片的样式
      searchQuery: '', // 添加搜索查询
    };
  },
  computed: {
    maxPage() {
      return Math.max(1, Math.ceil(this.totalNotes / this.pageSize));
    },
    notesWithPlaceholders() {
      const placeholders = Array.from({ length: this.pageSize - this.notes.length }, (_, index) => ({
        placeholder: true,
        id: `placeholder-${index}`,
      }));
      return [...this.notes, ...placeholders];
    },
    filteredNotes() {
      if (!this.searchQuery) {
        return this.notesWithPlaceholders;
      }
      
      const searchLower = this.searchQuery.toLowerCase();
      const filtered = this.notes.filter(note => 
        note.title?.toLowerCase().includes(searchLower) ||
        note.content?.toLowerCase().includes(searchLower)
      );
      
      if (filtered.length >= 3) {
        const placeholders = Array.from(
          { length: Math.max(0, this.pageSize - filtered.length) },
          (_, index) => ({
            placeholder: true,
            id: `placeholder-${index}`,
          })
        );
        return [...filtered, ...placeholders];
      }
      
      return filtered;
    },
    filteredNotesWithoutPlaceholders() {
      return this.filteredNotes.filter(note => !note.placeholder);
    },
  },
  methods: {
    fetchNotes() {
      const params = {
        page: this.currentPage,
        size: this.pageSize,
      };
      getNoteListService(params)
        .then(async (response) => {
          if (response && response.length > 0) {
            const detailedNotes = await Promise.all(
              response.map(async (note) => {
                const detailResponse = await getNoteDetailService({ id: note.id });
                return {
                  ...note,
                  ...detailResponse.data,
                };
              })
            );
            this.notes = detailedNotes;
            this.totalNotes = this.currentPage * this.pageSize + response.length;
            this.currentPageInput = this.currentPage;
          } else {
            ElMessage.warning("没有更多帖子了~");
            if (this.currentPage > 1) {
              this.currentPage--;
              this.currentPageInput = this.currentPage;
            }
          }
        })
        .catch((error) => {
          console.error("Error fetching note list:", error);
          ElMessage.error("Failed to fetch notes.");
        });
    },
    handlePageChange(page) {
      this.currentPage = page;
      this.currentPageInput = page;
      this.fetchNotes();
    },
    goToNoteDetail(noteId, userId) {
      if (!noteId) return;
      this.$router.push({ name: "NoteDetail", params: { id: noteId, userId } });
    },
    handleImageError(event) {
      event.target.src = "/assets/developer.png";
    },
    getImageStyle(note) {
      return {
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: 'translate(-50%, -50%)',
        maxWidth: '100%',
        maxHeight: '100%',
        width: 'auto',
        height: 'auto',
        objectFit: 'contain'
      }
    },
    handleSearch() {
      this.currentPage = 1;
    },
  },
  mounted() {
    this.fetchNotes();
  },
};
</script>

<style scoped>
.page-container {
  padding: 24px;
  min-height: 100vh;
  background-color: #f5f7fa;
}

.content-card {
  background-color: white;
  border-radius: 12px;
}

.note-card {
  cursor: pointer;
  transition: all 0.3s ease;
  border: none;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.note-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
}

.note-card:hover img {
  transform: scale(1.05);
}

.creator-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.placeholder-card {
  background-color: #f9fafb;
  border: 2px dashed #e5e7eb;
  opacity: 0.6;
  pointer-events: none;
}

.placeholder-card:hover {
  transform: none;
  box-shadow: none;
}

:deep(.el-pagination) {
  --el-pagination-hover-color: #409eff;
  --el-pagination-button-color: #606266;
  --el-pagination-button-disabled-color: #c0c4cc;
  --el-pagination-button-bg-color: #fff;
  --el-pagination-hover-bg-color: #ecf5ff;
}

:deep(.el-pagination .el-input__inner) {
  text-align: center;
}

.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 3rem;
}

.image-container {
  position: relative;
  width: 100%;
  border-radius: 12px;
  overflow: hidden;
  background-color: #f5f5f5;
}

.image-wrapper {
  position: relative;
  width: 100%;
  padding-top: 66.67%;
  overflow: hidden;
  background-color: #f5f5f5;
}

.note-image {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  min-width: 100%;
  min-height: 100%;
  object-fit: contain;
  transition: all 0.5s ease;
}

.image-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    to bottom,
    rgba(0, 0, 0, 0) 60%,
    rgba(0, 0, 0, 0.05) 100%
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.note-card:hover .note-image {
  transform: translate(-50%, -50%) scale(1.05);
}

.note-card:hover .image-overlay {
  opacity: 1;
}

.placeholder-card .image-container {
  background: linear-gradient(110deg, #eff1f3 8%, #e2e4e7 18%, #eff1f3 33%);
  background-size: 200% 100%;
  animation: shimmer 1.5s linear infinite;
}

@keyframes shimmer {
  to {
    background-position: -200% 0;
  }
}

@media (max-width: 640px) {
  .image-wrapper {
    padding-top: 75%;
  }
}

@media (min-width: 1024px) {
  .image-wrapper {
    padding-top: 56.25%;
  }
}

.image-wrapper::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: #f0f0f0;
}

.note-image {
  opacity: 1;
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.image-wrapper {
  background: linear-gradient(45deg, #f5f5f5 25%, #efefef 25%, #efefef 50%, #f5f5f5 50%, #f5f5f5 75%, #efefef 75%);
  background-size: 20px 20px;
}

.header-section {
  padding: 24px 0;
  margin-bottom: 24px;
  border-bottom: 1px solid #ebeef5;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
}

.title-section {
  flex-shrink: 0;
}

.search-bar {
  max-width: 300px;
  width: 100%;
}

.search-input :deep(.el-input__wrapper) {
  border-radius: 20px 0 0 20px;
  box-shadow: 0 0 0 1px #dcdfe6 inset;
}

.search-input :deep(.el-input-group__append) {
  border-radius: 0 20px 20px 0;
  background-color: #409eff;
  border-color: #409eff;
  padding: 0 20px;
}

.search-input :deep(.el-input-group__append .el-button) {
  color: white;
  border: none;
  margin: 0;
  padding: 0;
}

.search-input :deep(.el-input__inner) {
  height: 36px;
  line-height: 36px;
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    align-items: stretch;
  }

  .title-section {
    text-align: center;
    margin-bottom: 16px;
  }

  .search-bar {
    max-width: 100%;
  }
}

.no-results {
  padding: 40px 0;
  text-align: center;
}

@media (max-width: 640px) {
  .search-bar {
    max-width: 100%;
    padding: 0 16px;
  }
}

.empty-search-result {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  text-align: center;
}

.empty-image {
  width: 160px;
  height: 160px;
  object-fit: contain;
  margin-bottom: 24px;
  opacity: 0.6;
}

.empty-text {
  color: #909399;
  font-size: 16px;
  margin: 0;
}

@media (max-width: 640px) {
  .empty-image {
    width: 120px;
    height: 120px;
  }
  
  .empty-text {
    font-size: 14px;
  }
}

.grid {
  display: grid;
  gap: 24px;
}

@media (min-width: 768px) {
  .grid {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }
}
</style>