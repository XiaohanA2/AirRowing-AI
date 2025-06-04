<template>
  <div class="user-profile-container">
    <div class="user-profile">
      <!-- 优化背景图片展示区域 -->
      <div class="profile-header" :style="{ backgroundImage: `url(${userInfo.backgrounding || defaultBg})` }">
        <div class="profile-header__overlay">
          <div class="profile-header__content">
            <img :src="userInfo.avatar || defaultAvatar" alt="用户头像" class="profile-header__avatar" />
            <div class="profile-header__info">
              <h1 class="profile-header__name">{{ userInfo.nickname }}</h1>
              <p class="profile-header__id">@{{ userInfo.airrowingId || '未设置' }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 优化信息展示区域布局 -->
      <div class="info-grid">
        <div class="info-card">
          <div class="info-card__header">
            <i class="fas fa-user"></i>
            <h3>基本信息</h3>
          </div>
          <div class="info-card__content">
            <div class="info-item">
              <i class="fas fa-envelope"></i>
              <span class="info-label">邮箱</span>
              <span class="info-value">{{ userInfo.email }}</span>
            </div>
            <div class="info-item">
              <i class="fas fa-birthday-cake"></i>
              <span class="info-label">生日</span>
              <span class="info-value">{{ formattedBirthday }}</span>
            </div>
            <div class="info-item">
              <i class="fas fa-venus-mars"></i>
              <span class="info-label">性别</span>
              <span class="info-value">{{ gender }}</span>
            </div>
            <div class="info-item">
              <i class="fas fa-info-circle"></i>
              <span class="info-label">简介</span>
              <span class="info-value">{{ userInfo.introduction || '这个人很懒，什么都没写~' }}</span>
            </div>
          </div>
        </div>

        <!-- 修改数据统计卡片 -->
        <div class="info-card stats-card">
          <div class="info-card__header">
            <i class="fas fa-chart-bar"></i>
            <h3>个人数据</h3>
          </div>
          <div class="stats-grid">
            <div class="stat-item" v-for="(value, key) in statsData" :key="key">
              <div class="stat-icon">
                <i :class="value.icon"></i>
              </div>
              <div class="stat-content">
                <span class="stat-number">{{ calculateStat(key) }}</span>
                <span class="stat-label">{{ value.label }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 优化操作按钮 -->
      <div class="action-buttons">
        <button class="action-btn action-btn--primary" @click="openEditModal">
          <i class="fas fa-edit"></i>
          <span>编辑资料</span>
        </button>
        <button class="action-btn action-btn--secondary" @click="openChangePasswordModal">
          <i class="fas fa-key"></i>
          <span>修改密码</span>
        </button>
        <button class="action-btn action-btn--danger" @click="handleLogout">
          <i class="fas fa-sign-out-alt"></i>
          <span>退出登录</span>
        </button>
      </div>

      <!-- 其他模态窗口代码保持不变 -->
      <!-- 编辑信息模态窗口 -->
      <transition name="fade">
        <div v-if="isEditModalOpen" class="modal-overlay" @click.self="closeEditModal">
          <div class="modal">
            <h2>修改用户信息</h2>
            <form @submit.prevent="updateUserInfo" class="form">
              <div class="form__group">
                <label for="nickname">用户名:</label>
                <input id="nickname" type="text" v-model="userInfo.nickname" required />
              </div>
              <div class="form__group">
                <label for="airrowingId">用户标识:</label>
                <input 
                  id="airrowingId" 
                  type="text" 
                  v-model="userInfo.airrowingId" 
                  required 
                  placeholder="请输入用户标识"
                />
              </div>
              <div class="form__group">
                <label for="birthday">生日:</label>
                <input id="birthday" type="date" v-model="userInfo.birthday" required />
              </div>
              <div class="form__group">
                <label for="sex">性别:</label>
                <select id="sex" v-model="userInfo.sex" required>
                  <option value="1">女</option>
                  <option value="0">男</option>
                </select>
              </div>
              <div class="form__group">
                <label for="introduction">简介:</label>
                <textarea id="introduction" v-model="userInfo.introduction" rows="4"></textarea>
              </div>
              <div class="form__group">
                <label for="avatar">头像:</label>
                <input id="avatar" type="file" @change="onFileChange($event, 'avatar')" accept="image/*" />
                <img v-if="userInfo.avatar && isFile(userInfo.avatar)" :src="previewAvatar" alt="预览头像" class="form__preview" />
              </div>
              <div class="form__group">
                <label for="backgrounding">背景图片:</label>
                <input id="backgrounding" type="file" @change="onFileChange($event, 'backgrounding')" accept="image/*" />
                <img v-if="userInfo.backgrounding && isFile(userInfo.backgrounding)" :src="previewBackground" alt="预览背景" class="form__preview" />
              </div>
              <div class="form__actions">
                <button type="submit" class="btn btn--save" :disabled="loading">保存修改</button>
                <button type="button" class="btn btn--cancel" @click="closeEditModal">取消</button>
              </div>
            </form>
          </div>
        </div>
      </transition>

      <!-- 修改密码模态窗口 -->
      <transition name="fade">
        <div v-if="isChangePasswordModalOpen" class="modal-overlay" @click.self="closeChangePasswordModal">
          <div class="modal">
            <h2>修改密码</h2>
            <form @submit.prevent="onSubmit" ref="formRef" class="form">
              <div class="form__group">
                <label for="email">邮箱:</label>
                <input id="email" type="email" v-model="passwordData.email" required />
              </div>
              <div v-if="!isCodeSent" class="form__group">
                <button type="button" class="btn btn--send-code" @click="handleSendCode" :disabled="loading">发送验证码</button>
              </div>
              <div v-else class="form__group">
                <p>验证码已发送，{{ countdown }} 秒后可重新发送</p>
                <label for="code">验证码:</label>
                <input id="code" type="text" v-model="passwordData.code" required />
              </div>
              <div class="form__group">
                <label for="newPassword">新密码:</label>
                <el-input
                  v-model="passwordData.newPassword"
                  :type="passwordVisible ? 'text' : 'password'"
                  placeholder="请输入新密码"
                  show-password
                  required
                />
              </div>
              <div class="form__group">
                <label for="confirmNewPassword">确认新密码:</label>
                <el-input
                  v-model="passwordData.confirmNewPassword"
                  :type="passwordVisible ? 'text' : 'password'"
                  placeholder="请确认新密码"
                  show-password
                  required
                />
              </div>
              <div class="form__actions">
                <button type="submit" class="btn btn--update-password" :disabled="loading">修改密码</button>
                <button type="button" class="btn btn--cancel" @click="closeChangePasswordModal">取消</button>
              </div>
            </form>
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import { getToken } from "@/composables/cookie";
import { sendVerificationCode, login } from '@/api/admin/user'; // 引入 login 方法
import { ref, reactive, computed, onMounted } from 'vue';
import { useUserStore } from '@/stores/user'
import { useRouter } from 'vue-router'
import { showMessage, showModel } from '@/composables/util'

export default {
  setup() {
    const userInfo = reactive({
      id: null,
      nickname: "",
      airrowingId: "",
      avatar: "",
      email: "",
      birthday: "",
      sex: "",
      backgrounding: "",
      introduction: ""
    });

    const userStats = reactive({
        fansTotal: "0",
        followingTotal: "0",
        noteTotal: "0",
        likeTotal: "0",
        collectTotal: "0"
    });

    const passwordData = reactive({
      email: "",
      code: "",
      newPassword: "",
      confirmNewPassword: "",
    });

    const isEditing = ref(false);
    const isChangePassword = ref(false);
    const isEditModalOpen = ref(false);
    const isChangePasswordModalOpen = ref(false);
    const isCodeSent = ref(false);
    const loading = ref(false);
    const countdown = ref(0);
    const formRef = ref(null);
    const passwordVisible = ref(false);

    const userStore = useUserStore()
    const router = useRouter()

    // 计算属性：格式化生日
    const formattedBirthday = computed(() => {
      if (!userInfo.birthday) return '';
      const date = new Date(userInfo.birthday);
      return date.toLocaleDateString();
    });
    // 计算属性：性别显示
    const gender = computed(() => {
        return userInfo.sex === 1 ? '女' : '男';
    });

    // 预览头像
    const previewAvatar = computed(() => {
      return isFile(userInfo.avatar) ? URL.createObjectURL(userInfo.avatar) : userInfo.avatar;
    });

    // 预览背景图片
    const previewBackground = computed(() => {
      return isFile(userInfo.backgrounding) ? URL.createObjectURL(userInfo.backgrounding) : userInfo.backgrounding;
    });

    // 判断是否为文件
    const isFile = (file) => {
      return file instanceof File;
    };

    // 获取用户信息
    const fetchUserInfo = async () => {
      try {
        const token = getToken();
        const response = await axios.get("/api/user/user/current", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (response.data.success) {
          Object.assign(userInfo, response.data.data);
          // 在成功获取用户信息后，立即获取用户统计信息
          if (userInfo.id) {
            await fetchUserStats(userInfo.id);
          }
        } else {
          console.error("获取用户信息失败:", response.data.message);
          alert("获取用户信息失败: " + response.data.message);
        }
      } catch (error) {
        console.error("获取用户信息出错:", error);
        alert("获取用户信息出错，请稍后重试。");
      }
    };

    const fetchUserStats = async (id) => {
        try {
            const token = getToken();          
            console.log("正在获取用户统计信息，用户ID:", id);
            const response = await axios.post(`/api/count/count/user`, 
            { userId: id }, 
            {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.data.success) {
                Object.assign(userStats, response.data.data);
                console.log("用户统计信息:", userStats);
            } else {
                console.error("获取用户统计信息失败:", response.data.message);
            }
        } catch (error) {
            console.error("完整的错误对象:", error);
            if (error.response) {
                console.error("错误响应:", {
                    status: error.response.status,
                    data: error.response.data,
                    headers: error.response.headers
                });
            }
        }
    };

    // 发送验证码
    const handleSendCode = async () => {
      if (!passwordData.email) {
        alert("请先输入邮箱");
        return;
      }
      try {
        loading.value = true;
        await sendVerificationCode(passwordData.email);
        alert("验证码已发送，请查收邮箱");
        isCodeSent.value = true;
        countdown.value = 60;
        const interval = setInterval(() => {
          countdown.value--;
          if (countdown.value === 0) {
            clearInterval(interval);
            isCodeSent.value = false;
          }
        }, 1000);
      } catch (error) {
        console.error("发送验证码出错:", error);
        alert(error.response?.data?.message || "发送验证码失败");
      } finally {
        loading.value = false;
      }
    };

    // 密码验证函数
    const validatePassword = (password) => {
      const lengthValid = password.length >= 6 && password.length <= 20;
      const pattern = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,20}$/;
      return lengthValid && pattern.test(password);
    };

    // 提交修改密码表单
    const onSubmit = async () => {
      if (!passwordData.email) {
        alert("邮箱不能为空");
        return;
      }
      if (!isCodeSent.value) {
        alert("请先发送验证码");
        return;
      }
      if (!passwordData.code) {
        alert("验证码不能为空");
        return;
      }
      if (!passwordData.newPassword || !passwordData.confirmNewPassword) {
        alert("密码不能为空");
        return;
      }
      if (passwordData.newPassword !== passwordData.confirmNewPassword) {
        alert("新密码与确认密码不一致");
        return;
      }
      if (!validatePassword(passwordData.newPassword)) {
        alert("密码应为6到20个字符，且包含至少一个字母和一个数字");
        return;
      }

      try {
        loading.value = true;
        const { email, code, newPassword } = passwordData;

        // 验证验证码
        const loginResponse = await login(email, '', code, 1); // 1 表示验证码登录

        if (loginResponse.success) {
          const token = getToken();

          await axios.post(
            "/api/auth/password/update",
            { newPassword },
            {
              headers: {
                Authorization: `Bearer ${token}`,
              },
            }
          );

          alert("密码已修改");
          resetPasswordForm();
          closeChangePasswordModal();
        } else {
          alert(loginResponse.message || "验证码验证失败");
        }
      } catch (error) {
        console.error("修改密码出错:", error);
        alert(error.response?.data?.message || "密码修改失败");
      } finally {
        loading.value = false;
      }
    };

    // 重置密码表单
    const resetPasswordForm = () => {
      passwordData.email = "";
      passwordData.code = "";
      passwordData.newPassword = "";
      passwordData.confirmNewPassword = "";
      isCodeSent.value = false;
    };

    // 更新用户信息
    const updateUserInfo = async () => {
      try {
        loading.value = true;
        const token = getToken();
        const formData = new FormData();
        formData.append("nickname", userInfo.nickname);
        formData.append("birthday", userInfo.birthday);
        formData.append("sex", userInfo.sex);
        formData.append("introduction", userInfo.introduction);
        formData.append("airrowingId", userInfo.airrowingId);
        if (isFile(userInfo.avatar)) {
          formData.append("avatar", userInfo.avatar);
        }

        if (isFile(userInfo.backgrounding)) {
          formData.append("backgrounding", userInfo.backgrounding);
        }

        const response = await axios.post(
          "/api/user/user/update",
          formData,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "multipart/form-data"
            },
          }
        );

        if (response.data.success) {
          alert("用户信息已更新");
          closeEditModal();
          fetchUserInfo(); // 重新获取更新后的用户信息
          fetchUserStats(userInfo.id); 
        } else {
          console.error("更新用户信息失败:", response.data.message);
          alert("更新用户信息失败: " + response.data.message);
        }
      } catch (error) {
        console.error("更新用户信息出错:", error);
        alert(error.response?.data?.message || "用户信息更新失败");
      } finally {
        loading.value = false;
      }
    };

    // 处理文件选择
    const onFileChange = (event, field) => {
      const file = event.target.files[0];
      if (file) {
        userInfo[field] = file;
      }
    };

    // 打开编辑信息模态窗口
    const openEditModal = () => {
      isEditModalOpen.value = true;
    };

    // 关闭编辑信息模态窗口
    const closeEditModal = () => {
      isEditModalOpen.value = false;
      fetchUserInfo(); // 重新获取用户信息，重置表单
      fetchUserStats(userInfo.id); 
    };

    // 打开修改密码模态窗口
    const openChangePasswordModal = () => {
      isChangePasswordModalOpen.value = true;
    };

    // 关闭修改密码模态窗口
    const closeChangePasswordModal = () => {
      isChangePasswordModalOpen.value = false;
      resetPasswordForm(); // 重置密码表单
    };

    // 退出登录处理函数
    const handleLogout = async () => {
      const confirmed = await showModel('是否确认要退出登录？')
      if (confirmed) {
        const success = await userStore.logout()
        if (success) {
          showMessage('退出登录成功！')
          router.push('/login')
        }
      }
    }

    // 组件挂载时获取用户信息
    onMounted(() => {
      fetchUserInfo();
    });

    // 在 setup 中添加默认图片和统计数据配置
    const defaultAvatar = '/path/to/default-avatar.png';
    const defaultBg = '/path/to/default-bg.jpg';

    const statsData = {
      rowingYears: {
        label: '赛艇年数',
        icon: 'fas fa-ship',
        value: '3年'
      },
      memberDays: {
        label: '加入天数',
        icon: 'fas fa-calendar-alt',
        value: '180天'
      },
      noteTotal: {
        label: '发帖数',
        icon: 'fas fa-file-alt',
        // 不设置固定 value，使用 API 数据
      },
      likeTotal: {
        label: '获赞数',
        icon: 'fas fa-heart',
        // 不设置固定 value，使用 API 数据
      },
      trainingHours: {
        label: '训练时长',
        icon: 'fas fa-stopwatch',
        value: '256h'
      },
      level: {
        label: '用户等级',
        icon: 'fas fa-medal',
        value: 'Lv.8'
      }
    };

    // 修改计算方法，区分 API 数据和固定值
    const calculateStat = (key) => {
      // 如果是发帖数或获赞数，使用 API 数据
      if (key === 'noteTotal' || key === 'likeTotal') {
        return userStats[key] || '0';
      }
      // 其他统计项使用固定值
      return statsData[key].value;
    };

    // 在 setup 中返回这些新增的数据
    return {
      userInfo,
      passwordData,
      isEditing,
      isChangePassword,
      isEditModalOpen,
      isChangePasswordModalOpen,
      isCodeSent,
      loading,
      countdown,
      formRef,
      fetchUserInfo,
      updateUserInfo,
      onFileChange,
      handleSendCode,
      onSubmit,
      openEditModal,
      closeEditModal,
      openChangePasswordModal,
      closeChangePasswordModal,
      formattedBirthday,
      gender,
      previewAvatar,
      previewBackground,
      isFile,
      userStats,
      fetchUserStats,
      passwordVisible,
      defaultAvatar,
      defaultBg,
      statsData,
      calculateStat,
      handleLogout,
    };
  },
};
</script>

<style scoped>
.user-profile-container {
  display: flex;
  justify-content: center;
  min-height: calc(100vh - 20px);
  padding: 20px;
  box-sizing: border-box;
  background-color: #f0f2f5;
  overflow-y: auto;
  margin-top: 20px;
}

.user-profile {
  width: 100%;
  max-width: 1200px;
  background: linear-gradient(120deg, #ffffff, #f8f9fa, #ffffff);
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.profile-header {
  height: 300px;
  min-height: 200px;
  background-size: cover;
  background-position: center;
  border-radius: 15px;
  position: relative;
  margin-bottom: 25px;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

/* 主渐变背景 */
.profile-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    125deg,
    rgba(88, 156, 239, 0.6) 0%,     /* 清新蓝 */
    rgba(114, 192, 236, 0.5) 25%,    /* 天空蓝 */
    rgba(157, 185, 233, 0.5) 45%,    /* 淡紫蓝 */
    rgba(226, 189, 205, 0.5) 65%,    /* 柔和粉 */
    rgba(241, 208, 212, 0.4) 85%,    /* 浅粉 */
    rgba(246, 220, 217, 0.4) 100%    /* 米白粉 */
  );
  z-index: 1;
  backdrop-filter: blur(5px);
  transition: all 0.3s ease;
}

/* 动态光效层 */
.profile-header::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: 
    radial-gradient(
      circle at 20% 50%,
      rgba(255, 255, 255, 0.15) 0%,
      transparent 50%
    ),
    radial-gradient(
      circle at 80% 50%,
      rgba(255, 255, 255, 0.12) 0%,
      transparent 50%
    );
  z-index: 2;
  opacity: 0.8;
  transition: opacity 0.3s ease;
}

/* 信息展示区域的渐变遮罩 */
.profile-header__overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 30px 25px;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    rgba(255, 255, 255, 0.1) 20%,
    rgba(255, 255, 255, 0.95) 100%
  );
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  display: flex;
  align-items: center;
  gap: 20px;
  z-index: 3;
}

/* 头像样式优化 */
.profile-header__avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: 4px solid rgba(255, 255, 255, 0.95);
  box-shadow: 
    0 4px 12px rgba(0, 0, 0, 0.1),
    0 0 0 4px rgba(255, 255, 255, 0.4);
  transition: all 0.3s ease;
  z-index: 4;
}

.profile-header__avatar:hover {
  transform: scale(1.05);
  box-shadow: 
    0 6px 16px rgba(0, 0, 0, 0.15),
    0 0 0 4px rgba(255, 255, 255, 0.5);
}

/* 用户名样式优化 */
.profile-header__name {
  color: #2c3e50;
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-shadow: 0 2px 4px rgba(255, 255, 255, 0.3);
  z-index: 4;
}

/* 悬停效果 */
.profile-header:hover::before {
  background: linear-gradient(
    125deg,
    rgba(88, 156, 239, 0.7) 0%,
    rgba(114, 192, 236, 0.6) 25%,
    rgba(157, 185, 233, 0.6) 45%,
    rgba(226, 189, 205, 0.6) 65%,
    rgba(241, 208, 212, 0.5) 85%,
    rgba(246, 220, 217, 0.5) 100%
  );
}

.profile-header:hover::after {
  opacity: 1;
}

/* 信息展示区域使用网格布局并限制高度 */
.info-grid {
  height: 65%;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
  margin-bottom: 15px;
}

.info-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 10px;
  box-shadow: 
    0 4px 6px rgba(0, 0, 0, 0.05),
    0 1px 3px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.8);
}

.info-card:hover {
  transform: translateY(-5px) scale(1.01);
  box-shadow: 
    0 12px 20px rgba(0, 0, 0, 0.08),
    0 4px 8px rgba(0, 0, 0, 0.12);
}

.info-card__header {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-bottom: 2px solid rgba(0, 123, 255, 0.1);
  padding: 12px 18px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-radius: 10px 10px 0 0;
}

.info-card__header h3 {
  font-weight: 600;
  color: #2c3e50;
  letter-spacing: 0.5px;
}

.info-card__content {
  padding: 15px;
  flex: 1;
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 123, 255, 0.3) transparent;
}

.info-card__content p {
  color: #495057;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  margin: 0;
  padding: 12px 8px;
}

.info-card__content p:last-child {
  border-bottom: none;
}

.info-card__content p i {
  color: #007bff;
  width: 20px;
  text-align: center;
}

.info-card__stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 15px;
}

.stat-item {
  background: rgba(0, 123, 255, 0.03);
  border-radius: 12px;
  padding: 15px 10px;
  border: 1px solid rgba(0, 123, 255, 0.08);
  text-align: center;
  transition: background-color 0.3s ease;
}

.stat-item:hover {
  background-color: rgba(0, 123, 255, 0.1);
}

.stat-number {
  font-size: 22px;
  font-weight: 700;
  background: linear-gradient(135deg, #0056b3, #007bff);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  display: block;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 13px;
  color: #6c757d;
  font-weight: 500;
}

/* 底部按钮区域 */
.action-buttons {
  height: 15%;
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
}

.action-btn {
  padding: 10px 24px;
  border-radius: 20px;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85em;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.action-btn::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(255, 255, 255, 0.2);
  transform: translateX(-100%);
  transition: transform 0.3s ease;
}

.action-btn:hover::after {
  transform: translateX(0);
}

.action-btn--primary {
  background: linear-gradient(135deg, #0056b3, #007bff);
  color: white;
  border: none;
}

.action-btn--primary:hover {
  background: linear-gradient(135deg, #004494, #0056b3);
  transform: translateY(-2px);
}

.action-btn--secondary {
  background: linear-gradient(135deg, #495057, #6c757d);
  color: white;
  border: none;
}

.action-btn--secondary:hover {
  background: linear-gradient(135deg, #404850, #495057);
  transform: translateY(-2px);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: 1fr;
    height: auto;
  }

  .info-card__stats {
    grid-template-columns: repeat(2, 1fr);
  }

  .action-buttons {
    flex-direction: row;
    padding: 10px 0;
  }
}

/* 添加一些动画效果 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.info-card {
  animation: fadeIn 0.5s ease forwards;
}

/* 添加滚动条美化 */
.info-card__content {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 123, 255, 0.3) transparent;
}

.info-card__content::-webkit-scrollbar {
  width: 6px;
}

.info-card__content::-webkit-scrollbar-track {
  background: transparent;
}

.info-card__content::-webkit-scrollbar-thumb {
  background-color: rgba(0, 123, 255, 0.3);
  border-radius: 3px;
}

/* 添加加载动画 */
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.loading {
  background: linear-gradient(
    90deg,
    #f0f0f0 25%,
    #f8f8f8 50%,
    #f0f0f0 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

/* 添加内容切换动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(10px);
}

/* 模态窗口基础样式优化 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(8px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  animation: fadeIn 0.3s ease;
}

.modal {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 20px;
  padding: 30px;
  width: 90%;
  max-width: 550px;
  max-height: 85vh;
  overflow-y: auto;
  box-shadow: 
    0 15px 35px rgba(0, 0, 0, 0.1),
    0 3px 10px rgba(0, 0, 0, 0.07),
    0 0 0 1px rgba(255, 255, 255, 0.1);
  animation: modalSlideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
}

/* 模态窗口标题样式 */
.modal h2 {
  color: #2c3e50;
  font-size: 1.8em;
  margin-bottom: 25px;
  padding-bottom: 15px;
  border-bottom: 2px solid rgba(0, 123, 255, 0.1);
  position: relative;
  text-align: center;
}

.modal h2::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 50%;
  transform: translateX(-50%);
  width: 100px;
  height: 2px;
  background: linear-gradient(90deg, #007bff, #00d2ff);
}

/* 表单组样式优化 */
.form__group {
  position: relative;
  margin-bottom: 20px;
}

.form__group label {
  display: block;
  margin-bottom: 8px;
  color: #495057;
  font-weight: 600;
  font-size: 0.95em;
  transition: all 0.3s ease;
}

.form__group input,
.form__group select,
.form__group textarea {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e9ecef;
  border-radius: 12px;
  font-size: 1em;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.9);
}

.form__group input:hover,
.form__group select:hover,
.form__group textarea:hover {
  border-color: #adb5bd;
}

.form__group input:focus,
.form__group select:focus,
.form__group textarea:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.15);
  outline: none;
}

/* 文件上传样式优化 */
.form__group input[type="file"] {
  padding: 10px;
  background: #f8f9fa;
  border: 2px dashed #dee2e6;
  cursor: pointer;
  transition: all 0.3s ease;
}

.form__group input[type="file"]:hover {
  background: #e9ecef;
  border-color: #adb5bd;
}

/* 预览图片样式 */
.form__preview {
  max-width: 120px;
  max-height: 120px;
  margin-top: 15px;
  border-radius: 12px;
  object-fit: cover;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
}

.form__preview:hover {
  transform: scale(1.05);
}

/* 按钮样式优化 */
.form__actions {
  display: flex;
  gap: 15px;
  justify-content: flex-end;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.btn {
  padding: 12px 24px;
  border-radius: 12px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.95em;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.btn::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 300px;
  height: 300px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  transform: translate(-50%, -50%) scale(0);
  transition: transform 0.5s ease;
}

.btn:active::after {
  transform: translate(-50%, -50%) scale(1);
}

.btn--save {
  background: linear-gradient(135deg, #007bff, #0056b3);
  color: white;
  box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
}

.btn--save:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
}

.btn--cancel {
  background: linear-gradient(135deg, #6c757d, #495057);
  color: white;
  box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
}

.btn--cancel:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4);
}

.btn--send-code {
  background: linear-gradient(135deg, #28a745, #218838);
  color: white;
  box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
}

.btn--send-code:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
}

/* 加载状态样式 */
.btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none !important;
}

/* 动画效果 */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes modalSlideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 滚动条美化 */
.modal::-webkit-scrollbar {
  width: 8px;
}

.modal::-webkit-scrollbar-track {
  background: transparent;
}

.modal::-webkit-scrollbar-thumb {
  background: rgba(0, 123, 255, 0.2);
  border-radius: 4px;
}

.modal::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 123, 255, 0.3);
}

/* 响应式优化 */
@media (max-width: 576px) {
  .modal {
    width: 95%;
    padding: 20px;
    margin: 10px;
  }

  .form__actions {
    flex-direction: column;
  }

  .btn {
    width: 100%;
    margin-bottom: 10px;
  }
  
  .modal h2 {
    font-size: 1.5em;
  }
}

/* 验证码倒计时样式 */
.countdown {
  color: #6c757d;
  font-size: 0.9em;
  margin-top: 8px;
  text-align: center;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* 错误提示样式 */
.form__error {
  color: #dc3545;
  font-size: 0.85em;
  margin-top: 5px;
  display: flex;
  align-items: center;
  gap: 5px;
  animation: shakeError 0.6s;
}

@keyframes shakeError {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px); }
  75% { transform: translateX(5px); }
}

/* 成功提示样式 */
.form__success {
  color: #28a745;
  font-size: 0.85em;
  margin-top: 5px;
  display: flex;
  align-items: center;
  gap: 5px;
  animation: fadeInSuccess 0.3s;
}

@keyframes fadeInSuccess {
  from { opacity: 0; transform: translateY(-5px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 添加 Element Plus 输入框样式覆盖 */
.form__group :deep(.el-input__wrapper) {
  border-radius: 12px;
  padding: 0;
}

.form__group :deep(.el-input__inner) {
  height: 42px;
  padding: 0 16px;
}

.form__group :deep(.el-input__suffix) {
  padding-right: 12px;
}

/* 优化卡片样式 */
.info-item {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  transition: background-color 0.3s ease;
}

.info-item:hover {
  background-color: rgba(0, 123, 255, 0.05);
}

.info-item i {
  color: #007bff;
  width: 24px;
  margin-right: 12px;
}

.info-label {
  color: #6c757d;
  width: 60px;
  margin-right: 12px;
  font-weight: 500;
}

.info-value {
  color: #2c3e50;
  flex: 1;
}

/* 优化统计卡片布局 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  padding: 20px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px;
  background: rgba(0, 123, 255, 0.03);
  border-radius: 12px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #007bff, #00d2ff);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-2px);
  background: rgba(0, 123, 255, 0.08);
}

.stat-item:hover::before {
  opacity: 1;
}

.stat-icon {
  font-size: 24px;
  color: #007bff;
  margin-bottom: 8px;
  transition: transform 0.3s ease;
}

.stat-item:hover .stat-icon {
  transform: scale(1.1);
}

.stat-content {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 24px;
  font-weight: 700;
  color: #2c3e50;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #6c757d;
}

/* 优化头部信息区域 */
.profile-header__content {
  display: flex;
  align-items: center;
  gap: 20px;
}

.profile-header__info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.profile-header__id {
  color: #6c757d;
  font-size: 14px;
  margin: 0;
}

/* 响应式优化 */
@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .info-item {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .info-label {
    margin-bottom: 4px;
  }
}

@media (max-width: 480px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    padding: 10px;
  }
  
  .stat-item {
    padding: 12px;
  }
  
  .stat-number {
    font-size: 20px;
  }
  
  .stat-label {
    font-size: 12px;
  }
}

/* 添加退出登录按钮样式 */
.action-btn--danger {
  background: linear-gradient(135deg, #dc3545, #c82333);
  color: white;
  border: none;
}

.action-btn--danger:hover {
  background: linear-gradient(135deg, #c82333, #bd2130);
  transform: translateY(-2px);
}

/* 调整按钮间距 */
.action-buttons {
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: center;
  margin-top: 20px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .action-buttons {
    flex-direction: column;
    width: 100%;
  }

  .action-btn {
    width: 100%;
  }
}
</style>
