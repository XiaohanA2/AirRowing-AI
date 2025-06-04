import { createRouter, createWebHashHistory } from 'vue-router'
import Admin from '@/layouts/admin/admin.vue'

// 页面组件导入
import Login from '@/pages/login.vue'
import NoteList from '@/pages/note/list.vue'
import NoteDetail from '@/pages/note/detail.vue'
import TrainingAnalysis from '@/pages/training/analysis.vue'
import UserProfile from '@/pages/user/profile.vue'
import UserView from '@/pages/user/view.vue'
import Welcome from '@/pages/home/welcome.vue'
import Community from '@/pages/community/home.vue'

const routes = [
    {
        path: '/',
        component: Login,
        meta: {
            title: 'Airrowing 登录页'
        }
    },
    {
        path: '/login',
        component: Login,
        meta: {
            title: 'Airrowing 登录页'
        }
    },
    {
        path: "/main",
        component: Admin,
        children: [
            {
                path: "welcome",
                component: Welcome,
                meta: {
                    title: '首页'
                }
            },
            {
                path: "club",
                name: "Club",
                meta: { title: '俱乐部' },
                children: [
                    {
                        path: "list",
                        name: "ClubList",
                        component: () => import('@/pages/club/list.vue'),
                        meta: { title: '俱乐部列表' }
                    },
                    {
                        path: "my",
                        name: "MyClubs",
                        component: () => import('@/pages/club/my.vue'),
                        meta: { title: '我的俱乐部' }
                    },
                    {
                        path: "detail/:id",
                        name: "ClubDetail",
                        component: () => import('@/pages/club/detail.vue'),
                        meta: { title: '俱乐部详情' }
                    },
                    {
                        path: "activity/:clubId",
                        name: "ClubActivity",
                        component: () => import('@/pages/club/activity.vue'),
                        meta: { title: '俱乐部活动' }
                    }
                ]
            },
            {
                path: "user/profile",
                component: UserProfile,
                meta: {
                    title: '个人信息'
                }
            },
            {
                path: "note/detail/:id/:userId?",
                component: NoteDetail,
                name: "NoteDetail",
                meta: {
                    title: '帖子详情'
                }
            },
            {
                path: 'training/analysis',
                name: 'training-analysis',
                component: TrainingAnalysis,
                meta: {
                    title: '训练分析'
                }
            },
            {
                path: "note/list",
                component: NoteList,
                meta: {
                    title: '我的主页'
                }
            },
            {
                path: "community/home",
                component: Community,
                meta: {
                    title: '社区首页'
                }
            },
            {
                path: '/user/:userId',
                name: 'UserDetail',
                component: UserView,
                meta: {
                    title: '用户详情'
                }
            }
        ]
    },
]

const router = createRouter({
    history: createWebHashHistory(),
    routes,
})

router.beforeEach((to, from, next) => {
    // 如果还在使用旧路径，重定向到新路径
    if (to.path === '/admin/index') {
        next('/main/welcome')
        return
    }
    // ... 其他路由守卫逻辑 ...
    next()
})

export default router

