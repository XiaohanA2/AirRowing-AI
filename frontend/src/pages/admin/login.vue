<template>
    <div class="grid grid-cols-2 h-screen">
        <div class="col-span-2 order-2 p-10 md:col-span-1 md:order-1 bg-slate-900">
            <div
                class="flex justify-center items-center h-full flex-col animate__animated animate__bounceInLeft animate__fast">
                <h2 class="font-bold text-4xl mb-7 text-white">Airrownig</h2>
                <p class="text-white">一款赛艇运动训练与互动平台</p>
                <img src="@/assets/ins.png" class="w-1/2">
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
                            :prefix-icon="Lock" clearable show-password @input="validatePasswordInput" />
                        <div v-if="passwordValidation.show" class="password-validation-tips">
                            <div :class="['validation-item', passwordValidation.length ? 'valid' : 'invalid']">
                                长度在8-20个字符之间
                            </div>
                            <div :class="['validation-item', passwordValidation.uppercase ? 'valid' : 'invalid']">
                                包含大写字母
                            </div>
                            <div :class="['validation-item', passwordValidation.lowercase ? 'valid' : 'invalid']">
                                包含小写字母
                            </div>
                            <div :class="['validation-item', passwordValidation.number ? 'valid' : 'invalid']">
                                包含数字
                            </div>
                        </div>
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

// 添加密码验证状态
const passwordValidation = reactive({
    show: false,
    length: false,
    uppercase: false,
    lowercase: false,
    number: false
});

// 实时验证密码
const validatePasswordInput = (value) => {
    passwordValidation.show = true;
    passwordValidation.length = value.length >= 8 && value.length <= 20;
    passwordValidation.uppercase = /[A-Z]/.test(value);
    passwordValidation.lowercase = /[a-z]/.test(value);
    passwordValidation.number = /[0-9]/.test(value);
};

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
        { min: 8, max: 20, message: '密码长度应在 8 到 20 个字符之间', trigger: 'blur' },
        { pattern: /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z\d]{8,}$/, message: '密码必须包含大小写字母和数字', trigger: 'blur' }
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
                router.push('/admin/index')
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
/* 添加密码验证提示样式 */
.password-validation-tips {
    margin-top: 8px;
    padding: 8px;
    border-radius: 8px;
    background-color: #f8f9fa;
    font-size: 0.85em;
}

.validation-item {
    margin: 4px 0;
    padding-left: 20px;
    position: relative;
}

.validation-item::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 8px;
}

.validation-item.valid {
    color: #28a745;
}

.validation-item.invalid {
    color: #dc3545;
}

.validation-item.valid::before {
    background-color: #28a745;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z'/%3E%3C/svg%3E");
    background-size: 10px;
    background-position: center;
    background-repeat: no-repeat;
}

.validation-item.invalid::before {
    background-color: #dc3545;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z'/%3E%3C/svg%3E");
    background-size: 8px;
    background-position: center;
    background-repeat: no-repeat;
}
</style>