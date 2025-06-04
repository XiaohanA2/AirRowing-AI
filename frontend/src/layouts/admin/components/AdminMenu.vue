<template>
    <div class="menu-container"
        :style="{ width: menuStore.menuWidth }">
        <!-- 顶部 Logo, 指定高度为 64px, 和右边的 Header 头保持一样高 -->
        <div class="flex items-center justify-center h-[64px]">
            <img v-if="menuStore.menuWidth == '250px'" src="@/assets/base.png" class="h-[60px]">
            <img v-else src="@/assets/base2.png" class="h-[60px]">
        </div>

        <!-- 下方菜单 -->
        <el-menu :default-active="defaultActive" @select="handleSelect" :collapse="isCollapse"
            :collapse-transition="false">
            <template v-for="(item, index) in menus" :key="index">
                <!-- 处理有子菜单的情况 -->
                <template v-if="item.children">
                    <el-sub-menu :index="item.path">
                        <template #title>
                            <el-icon>
                                <component v-if="!item.isCustomIcon" :is="item.icon"></component>
                                <img v-else :src="item.icon" class="menu-icon">
                            </el-icon>
                            <span>{{ item.name }}</span>
                        </template>
                        <el-menu-item v-for="(child, childIndex) in item.children" 
                            :key="childIndex" 
                            :index="child.path">
                            <el-icon>
                                <component v-if="!child.isCustomIcon" :is="child.icon"></component>
                                <img v-else :src="child.icon" class="menu-icon">
                            </el-icon>
                            <span>{{ child.name }}</span>
                        </el-menu-item>
                    </el-sub-menu>
                </template>
                <!-- 处理没有子菜单的情况 -->
                <template v-else>
                    <el-menu-item :index="item.path">
                        <el-icon>
                            <component v-if="!item.isCustomIcon" :is="item.icon"></component>
                            <img v-else :src="item.icon" class="menu-icon">
                        </el-icon>
                        <span>{{ item.name }}</span>
                    </el-menu-item>
                </template>
            </template>
        </el-menu>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useMenuStore } from '@/stores/menu'

const menuStore = useMenuStore()

const route = useRoute()
const router = useRouter()

// 是否折叠
const isCollapse = computed(() => !(menuStore.menuWidth == '250px'))

// 根据路由地址判断哪个菜单被选中
const defaultActive = ref(route.path)

// 菜单选择事件
const handleSelect = (path) => {
    router.push(path)
}

const menus = [
    {
        'name': '首页',
        'icon': 'HomeFilled',
        'isCustomIcon': false,
        'path': '/main/welcome'
    },
    {
        'name': '俱乐部',
        'icon': new URL('@/assets/icons/club.png', import.meta.url).href,
        'isCustomIcon': true,
        'path': '/main/club/list',
    },
    {
        'name': '赛艇社区',
        'icon': new URL('@/assets/icons/community.png', import.meta.url).href,
        'isCustomIcon': true,
        'path': '/main/community',
        'children': [
            {
                'name': '社区首页',
                'icon': new URL('@/assets/icons/monitor.png', import.meta.url).href,
                'isCustomIcon': true,
                'path': '/main/community/home',
            },
            {
                'name': '我的主页',
                'icon': 'User',
                'isCustomIcon': false,
                'path': '/main/note/list',
            },
        ]
    },
    {
        'name': '训练分析',
        'icon': 'DataAnalysis',
        'isCustomIcon': false,
        'path': '/main/training/analysis'
    },
    {
        'name': '个人信息',
        'icon': 'setting',
        'isCustomIcon': false,
        'path': '/main/user/profile',
    }
]
</script>

<style>
.el-menu {
    background-color: rgb(30 41 59 / 1);
    border-right: 0;
}

.el-sub-menu__title {
    color: #fff;
}

.el-sub-menu__title:hover {
    background-color: #ffffff10;
}


.el-menu-item.is-active {
    background-color: #409eff10;
    color: #fff;
}

.el-menu-item.is-active:before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 2px;
    height: 100%;
    background-color: var(--el-color-primary);
}

.el-menu-item {
    color: #fff;
}

.el-menu-item:hover {
    background-color: #ffffff10;
}

/* 添加子菜单样式 */
.el-sub-menu .el-menu-item {
    background-color: rgb(15, 23, 42) !important;
}

.el-sub-menu .el-menu-item:hover {
    background-color: #ffffff15 !important;
}

.el-sub-menu .el-menu-item.is-active {
    background-color: #409eff15 !important;
}

/* 添加滚动控制 */
.menu-container {
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    background-color: rgb(30 41 59);
    color: white;
    transition: all 0.3s ease;
    z-index: 1000;
    overflow-y: auto;
    -ms-overflow-style: none;
    scrollbar-width: none;
}

/* 隐藏滚动条 */
.menu-container::-webkit-scrollbar {
    display: none;
}

/* 添加菜单图标样式 */
.menu-icon {
    width: 20px;
    height: 20px;
    vertical-align: middle;
}

/* 确保折叠时图标居中 */
.el-menu--collapse .menu-icon {
    margin-right: 0;
}
</style>