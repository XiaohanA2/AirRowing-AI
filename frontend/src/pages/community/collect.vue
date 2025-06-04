<template>
  <el-card class="page-container">
    <div class="scroll-content" @scroll="handleScroll">
      <el-row :gutter="20">
        <el-col v-for="(article, index) in articles.slice(0, displayedItems)" :key="article.id" :span="6">
          <el-card shadow="hover" class="box-card" @click="openDrawer2(article.id, article)" style="cursor: pointer;">
            <div class="image-container">
              <img src="/assets/developer.png" alt="默认图片"
                class="default-image" />
              <img v-if="article.imgUris" :src="article.imgUris" alt=""
                class="overlay-image" />
            </div>

            <div style="margin-top: 10px; font-weight: bold;">
              {{ article.title }}
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </el-card>

</template>

<script setup>
import { ref, onMounted } from 'vue'
import { addCommentService, subCommentService, deleteCommentService, noteCommentCountService, noteFirstCommentService, collectListService, collectListDetailService, doCollectService, unCollectService, dolikeNoteService, nolikeNoteService, likeListService, noteCountService } from '@/api/collect.js'
import { noFollowService, doFollowService, followService, currentUserService } from '@/api/follow.js'
import { useRouter } from 'vue-router';

//控制抽屉是否显示
const visibleDrawer = ref(false)

// Category and articles data
const articles = ref([
  /* {
    "id": 4,
    title: "陕西旅游攻略",
    categoryId: 3,
  } */
])

// Infinite scroll
const displayedItems = ref(10)
const handleScroll = (event) => {
  const { scrollTop, clientHeight, scrollHeight } = event.target
  if (scrollTop + clientHeight >= scrollHeight - 50) {
    displayedItems.value += 10
  }
}


const size = ref(1000)//收藏列表请求参数
const page = ref(1)//收藏列表请求参数
const userId = ref(6101) // 用户ID

const currentUser = async () => {//获取当前用户ID
  let result = await currentUserService();
  /* console.log(result) */
  userId.value = result.data.id;
}

const collectList = async () => {//获取收藏列表相关
  let params = {
    size: size.value,
    userId: userId.value,
    page: page.value
  }
  let result = await collectListService(params);

  articles.value = result.data;
  // 为了并行处理多个收藏帖子详情��请求，使用 Promise.all
  const updatedArticles = await Promise.all(
    articles.value.map(async (article) => {
      /* console.log(article.id); */
      let params = {
        id: article.id
      }
      const detailResult = await collectListDetailService(params);
      /* console.log(detailResult); */
      // 将 imgUris 属性附加到当前文章对象上
      return {
        ...article,
        imgUris: detailResult.data.imgUris
      };
    })
  );
  // 更新 articles.value
  articles.value = updatedArticles;
}

const router = useRouter();//新的帖子详情页面，把帖子id传过去
const openDrawer2 = (id) => {
  router.push({ name: 'NoteDetail', params: { id, userId: userId.value } });
};


const fetchAllData = async () => {
  await currentUser();
  await collectList();
};
onMounted(() => {
  fetchAllData(); // 组件加载时调用
});


</script>

<style scoped>
.page-container {
  min-height: 93%;
  box-sizing: border-box;
}

/* 搜索表单固定在顶部 */
.fixed-form {
  position: sticky;
  top: 0;
  background-color: #fff;
  padding: 10px;
  z-index: 10;
}

/* 无限滚动内容 */
.scroll-content {
  max-height: 70vh;
  overflow-y: auto;
}

/* 无限滚动内容 */
.scroll-drawercontent {
  max-height: 80vh;
  overflow-y: auto;
}


.avatar-uploader {
  :deep() {
    .avatar {
      width: 178px;
      height: 178px;
      display: block;
    }

    .el-upload {
      border: 1px dashed var(--el-border-color);
      border-radius: 6px;
      cursor: pointer;
      position: relative;
      overflow: hidden;
      transition: var(--el-transition-duration-fast);
    }

    .el-upload:hover {
      border-color: var(--el-color-primary);
    }

    .el-icon.avatar-uploader-icon {
      font-size: 28px;
      color: #8c939d;
      width: 178px;
      height: 178px;
      text-align: center;
    }
  }
}

.editor {
  width: 100%;

  :deep(.ql-editor) {
    min-height: 200px;
  }
}

.image-container {
    position: relative;
    height: 200px;
    width: 100%;
}

.default-image,
.overlay-image {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 100%;
    object-fit: cover;
    border-radius: 5px;
}

.overlay-image {
    z-index: 1;
}
</style>