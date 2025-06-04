<template>
    <div class="grid grid-cols-2 h-screen">
        <div class="col-span-2 order-2 md:col-span-1 md:order-1 bg-slate-900 relative overflow-hidden">
            <div class="absolute inset-0">
                <img src="@/assets/ins.png" class="w-full h-full object-cover opacity-60" alt="background">
            </div>
            <div class="relative h-full flex flex-col justify-center items-center p-10 animate__animated animate__bounceInLeft animate__fast">
                <h2 class="font-bold text-5xl mb-4 text-white">AirRowing</h2>
                <p class="text-white text-xl mb-8">一款专注赛艇运动的训练与社交平台</p>
            </div>
        </div>
        <div class="col-span-2 order-1 md:col-span-1 md:order-2 bg-white">
            <div
                class="flex justify-center items-center h-full flex-col animate__animated animate__bounceInRight animate__fast">
                <h1 class="font-bold text-4xl mb-5">欢迎回来</h1>
                <div class="flex items-center justify-center mb-7 text-gray-400 space-x-2">
                    <span class="h-[1px] w-16 bg-gray-200"></span>
                    <span>{{ form.type === 2 ? '账号密码登录' : '如果是第一次登录，将会自动注册账号' }}</span>
                    <span class="h-[1px] w-16 bg-gray-200"></span>
                </div>
                <el-form class="w-5/6 md:w-2/5" ref="formRef" :rules="rules" :model="form">
                    <el-form-item prop="email">
                        <el-input size="large" v-model="form.email" placeholder="请输入邮箱" :prefix-icon="User" clearable />
                    </el-form-item>
                    <el-form-item v-if="form.type === 2" prop="password">
                        <el-input size="large" type="password" v-model="form.password" placeholder="请输入密码"
                            :prefix-icon="Lock" clearable show-password />
                    </el-form-item>
                    <el-form-item v-if="form.type === 1" prop="code">
                        <el-input size="large" v-model="form.code" placeholder="请输入验证码" :prefix-icon="Message"
                            clearable>
                            <template #append>
                                <el-button :disabled="countdown > 0" @click="sendCode">
                                    {{ countdown > 0 ? `${countdown}秒` : '发送验证码' }}
                                </el-button>
                            </template>
                        </el-input>
                    </el-form-item>

                    <el-form-item>
                        <el-button class="w-full mt-2" size="large" :loading="loading" type="primary"
                            @click="onSubmit">登录</el-button>
                    </el-form-item>
                    <div class="flex justify-center mt-2">
                        <el-button type="text" @click="toggleLoginType">{{ form.type === 2 ? '使用验证码登录' : '使用密码登录'
                            }}</el-button>
                    </div>
                </el-form>
            </div>
        </div>
    </div>
</template>

<script setup>
// 引入 Element Plus 中的用户、锁图标
import { User, Lock } from '@element-plus/icons-vue'
import { login, sendVerificationCode } from '@/api/admin/user'
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import { showMessage } from '@/composables/util'
import { setToken, getToken } from '@/composables/cookie'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

// 定义响应式的表单对象
const form = reactive({
    email: '',
    password: '',
    code: '',
    type: 2, // 默认密码登录
});

const router = useRouter();

// 登录按钮加载
const loading = ref(false);

// 表单引用
const formRef = ref(null);

// 验证码倒计时
const countdown = ref(0);

// 切换登录方式
const toggleLoginType = () => {
    form.type = form.type === 2 ? 1 : 2;
    form.password = ''; // 清除密码字段
    form.code = '';     // 清除验证码字段
};


// 表单验证规则
const rules = {
    email: [
        { required: true, message: '邮箱不能为空', trigger: 'blur' },
        { type: 'email', message: '请输入有效的邮箱地址', trigger: 'blur' }
    ],
    password: [
        { required: true, message: '密码不能为空', trigger: 'blur' },
        { min: 6, max: 20, message: '密码长度应在 6 到 20 个字符之间', trigger: 'blur' },
        { pattern: /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$/, message: '密码应包含至少一个字母和一个数字', trigger: 'blur' }
    ],
    code: [
        { required: true, message: '验证码不能为空', trigger: 'blur' },
        { len: 6, message: '验证码长度为6位', trigger: 'blur' }
    ]
};

const sendCode = async () => {
    if (!form.email) {
        showMessage('请先输入邮箱', 'warning');
        return;
    }

    try {
        await sendVerificationCode(form.email);
        showMessage('验证码已发送，请查收', 'success');
        countdown.value = 60;
        const interval = setInterval(() => {
            countdown.value--;
            if (countdown.value === 0) {
                clearInterval(interval);
            }
        }, 1000);
    } catch (error) {
        console.error("发送验证码错误：", error);
        showMessage(error.response?.data?.message || '发送验证码失败', 'error');
    }
};

const onSubmit = () => {
    console.log('登录')
    // 先验证 form 表单字段
    formRef.value.validate((valid) => {
        if (!valid) {
            console.log('表单验证不通过')
            return false
        }
        // 开始加载
        loading.value = true

        // 调用登录接口
        login(form.email, form.password, form.code, form.type).then((res) => {
            console.log(res)
            if (res.success == true) {
                showMessage('登录成功')

                // 存储 Token
                let token = res.data  // 直接获取 res.data 即可
                setToken(token)

                // 验证 Token 是否成功存储
                console.log('Token after setting:', getToken())

                // 获取用户信息，并存储到全局状态中
                userStore.setUserInfo()

                // 跳转到后台首页
                router.push('/main/welcome')
            } else {
                let message = res.message
                showMessage(message, 'error')
            }
        })
            .catch((error) => {
                console.error('登录请求出错：', error)
                showMessage('登录失败，请稍后重试', 'error')
            })
            .finally(() => {
                // 结束加载
                loading.value = false
            })
    })
}

// 按回车键后，执行登录事件
function onKeyUp(e) {
    console.log(e)
    if (e.key == 'Enter') {
        onSubmit()
    }
}

// 添加键盘监听
onMounted(() => {
    console.log('添加键盘监听')
    document.addEventListener('keyup', onKeyUp)
})

// 移除键盘监听
onBeforeUnmount(() => {
    document.removeEventListener('keyup', onKeyUp)
})


</script>

<style scoped>
/* 优化背景图片效果 */
.bg-slate-900 {
    position: relative;
    background: linear-gradient(to right, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.7));
}

.bg-slate-900::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        45deg,
        rgba(15, 23, 42, 0.9) 0%,
        rgba(15, 23, 42, 0.6) 100%
    );
    z-index: 1;
}

.relative {
    z-index: 2;
}

/* 图片动画效果 */
.object-cover {
    transition: transform 0.3s ease;
}

.bg-slate-900:hover .object-cover {
    transform: scale(1.05);
}

/* 响应式调整 */
@media (max-width: 768px) {
    .bg-slate-900 {
        min-height: 300px;
    }
}
</style>