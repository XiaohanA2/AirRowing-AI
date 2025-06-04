import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getUserInfo, logout as apiLogout } from '@/api/admin/user'
import { removeToken } from '@/composables/cookie'

export const useUserStore = defineStore('user', () => {
  // 用户信息
  const userInfo = ref({})

  // 设置用户信息
  function setUserInfo() {
    // 调用后头获取用户信息接口
    getUserInfo().then(res => {
      if (res.success == true) {
        userInfo.value = res.data
      }
    })
  }

  // 退出登录
  async function logout() {
    try {
      // 先清除本地存储
      removeToken();
      userInfo.value = {};
      
      // 再调用后端登出接口
      await apiLogout();
      return true;
    } catch (error) {
      console.warn("登出时发生错误:", error);
      // 即使后端请求失败,前端也已经清除了登录状态
      return true;
    }
  }

  return { userInfo, setUserInfo, logout };
},
  {
    // 开启持久化
    persist: true,
  }
)