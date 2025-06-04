<template>
    <!-- 设置背景色为白色、高度为 64px，padding-right 为 4， border-bottom 为 slate 100 -->
    <div class="header-container">
        <!-- 左边栏收缩、展开 -->
        <div class="w-[42px] h-[64px] cursor-pointer flex items-center justify-center text-gray-700 hover:bg-gray-200"
            @click="handleMenuWidth">
            <el-icon>
                <Fold v-if="menuStore.menuWidth == '250px'" />
                <Expand v-else />
            </el-icon>
        </div>

        <!-- 右边容器 -->
        <div class="ml-auto flex">
            <!-- 点击刷新页面 -->
            <el-tooltip class="box-item" effect="dark" content="刷新" placement="bottom">
                <div class="w-[42px] h-[64px] cursor-pointer flex items-center justify-center text-gray-700 hover:bg-gray-200"
                    @click="handleRefresh">
                    <el-icon>
                        <Refresh />
                    </el-icon>
                </div>
            </el-tooltip>

            <!-- 点击全屏展示 -->
            <el-tooltip class="box-item" effect="dark" content="全屏" placement="bottom">
                <div class="w-[42px] h-[64px] cursor-pointer flex items-center justify-center text-gray-700 mr-2 hover:bg-gray-200"
                    @click="toggle">
                    <el-icon>
                        <FullScreen v-if="!isFullscreen" />
                        <Aim v-else />
                    </el-icon>
                </div>
            </el-tooltip>

            <!-- 登录用户头像 -->
            <el-dropdown class="flex items-center justify-center" @command="handleCommand">
                <span class="el-dropdown-link flex items-center justify-center text-gray-700 text-xs">
                    <!-- 头像 Avatar -->
                    <el-avatar class="mr-2" :size="25"
                        src="https://img.quanxiaoha.com/quanxiaoha/f97361c0429d4bb1bc276ab835843065.jpg" />
                    {{ userStore.userInfo.username }}
                    <el-icon class="el-icon--right">
                        <arrow-down />
                    </el-icon>
                </span>
                <template #dropdown>
                    <el-dropdown-menu>
                        <el-dropdown-item command="logout">退出登录</el-dropdown-item>
                    </el-dropdown-menu>
                </template>
            </el-dropdown>
        </div>
    </div>

</template>

<script setup>
import { ref, reactive, watch } from 'vue'
import { useMenuStore } from '@/stores/menu'
import { useUserStore } from '@/stores/user'
import { useFullscreen } from '@vueuse/core'
import { updateAdminPassword } from '@/api/admin/user'
import { showMessage, showModel } from '@/composables/util'
import { useRouter } from 'vue-router'

const router = useRouter()

// isFullscreen 表示当前是否处于全屏；toggle 用于动态切换全屏、非全屏
const { isFullscreen, toggle } = useFullscreen()

// 引入了菜单 Store
const menuStore = useMenuStore()
// 引入了用户 Store
const userStore = useUserStore()

// icon 点击事件
const handleMenuWidth = () => {
    menuStore.handleMenuWidth()
}

// 刷新页面
const handleRefresh = () => location.reload()

// 对话框是否显示
const dialogVisible = ref(false)

// 下拉菜单事件处理
const handleCommand = (command) => {
    // 更新密码
    if (command == 'updatePassword') {
        // 显示修改密码对话框
        dialogVisible.value = true
    } else if (command == 'logout') { // 退出登录
        logout()
    }
}

// 退出登录
async function logout() {
    const confirmed = await showModel('是否确认要退出登录？');
    if (confirmed) {
        const success = await userStore.logout();
        if (success) {
            showMessage('退出登录成功！');
            router.push('/login');
        }
    }
}

// 表单引用
const formRef = ref(null)

// 修改用户密码表单对象
const form = reactive({
    username: userStore.userInfo.username || '',
    password: '',
    rePassword: ''
})

// 监听Pinia store中的某个值的变化
watch(() => userStore.userInfo.username, (newValue, oldValue) => {
      // 在这里处理变化后的值
      console.log('新值:', newValue);
      console.log('旧值:', oldValue);
      
      // 可以在这里执行任何你需要的逻辑
      // 重新将新的值，设置会 form 对象中
      form.username = newValue
});

// 规则校验
const rules = {
    username: [
        {
            required: true,
            message: '用户名不能为空',
            trigger: 'blur'
        }
    ],
    password: [
        {
            required: true,
            message: '密码不能为空',
            trigger: 'blur',
        },
    ],
    rePassword: [
        {
            required: true,
            message: '确认密码不能为空',
            trigger: 'blur',
        },
    ]
}

const onSubmit = () => {
    // 先验证 form 表单字段
    formRef.value.validate((valid) => {
        if (!valid) {
            console.log('表单验证不通过')
            return false
        }

        if (form.password != form.rePassword) {
            showMessage('两次密码输入不一致，请检查！', 'warning')
            return
        }

        // 调用修改用户密码接口
        updateAdminPassword(form).then((res) => {
            console.log(res)
            // 判断是否成功
            if (res.success == true) {
                showMessage('密码重置成功，请重新登录！')
                // 退出登录
                userStore.logout()

                // 隐藏对话框
                dialogVisible.value = false

                // 跳转登录页
                router.push('/login')
            } else {
                // 获取服务端返回的错误消息
                let message = res.message
                // 提示消息
                showMessage(message, 'error')
            }
        })
    })
}

</script>

<style scoped>
.header-container {
    position: fixed;
    top: 0;
    right: 0;
    left: 250px;
    height: 64px;
    background-color: white;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    padding-right: 1rem;
    z-index: 40;
    transition: all 0.3s ease;
}

.menu-collapsed .header-container {
    left: 64px;
}
</style>
