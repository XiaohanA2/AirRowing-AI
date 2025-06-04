<template>
    <div class="min-h-screen">
        <div class="wrapper" :class="{ 'menu-collapsed': isCollapse }">
            <!-- 左侧菜单 -->
            <AdminMenu />
            
            <!-- 右侧内容 -->
            <div class="content-container">
                <!-- 头部导航 -->
                <AdminHeader />
                
                <!-- 标签导航栏 -->
                <div class="tag-list-container fixed left-0 right-0 z-40" 
                     :style="{ top: '64px', marginLeft: menuStore.menuWidth }">
                    <AdminTagList />
                </div>
                
                <!-- 内容主体 -->
                <div class="main-content">
                    <router-view></router-view>
                </div>
                
                <!-- 底部版权 -->
                <AdminFooter />
            </div>
        </div>
    </div>
</template>

<script setup>
import { computed } from 'vue'
// 引入组件
import AdminFooter from './components/AdminFooter.vue';
import AdminHeader from './components/AdminHeader.vue';
import AdminMenu from './components/AdminMenu.vue';
import AdminTagList from './components/AdminTagList.vue';

import { useMenuStore } from '@/stores/menu'

const menuStore = useMenuStore()
const isCollapse = computed(() => menuStore.menuWidth === '64px')
</script>

<style scoped>
.el-header {
    padding: 0!important;
}

.el-footer {
    padding: 0!important;
}

/* 内容区域过渡动画：淡入淡出效果 */
/* 刚开始进入时 */
.fade-enter-from {
    /* 透明度 */
    opacity: 0;
}

/* 刚开始结束 */
.fade-enter-to {
    opacity: 1;
}

/* 刚开始离开 */
.fade-leave-from {
  opacity: 1;
}

/* 离开已结束 */
.fade-leave-to {
  opacity: 0;
}

/* 离开进行中 */
.fade-leave-active {
    transition: all 0.3s;
}

/* 进入进行中 */
.fade-enter-active {
    transition: all 0.3s;
    transition-delay: 0.3s;
}

.wrapper {
    position: relative;
    transition: all 0.3s ease;
}

.content-container {
    position: relative;
    margin-left: 250px;
    transition: all 0.3s ease;
    padding-top: 110px;
}

.menu-collapsed .content-container {
    margin-left: 64px;
}

.main-content {
    min-height: calc(100vh - 110px);
    padding: 0 20px;
}

.tag-list-container {
    background: white;
    border-bottom: 1px solid #e5e7eb;
    height: 44px;
    transition: all 0.3s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.menu-collapsed .tag-list-container {
    left: 64px;
}
</style>