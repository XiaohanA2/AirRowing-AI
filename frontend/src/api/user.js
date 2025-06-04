import axios from "@/axios";
import { getToken } from "@/composables/cookie"

// 获取登录用户信息
export function getUserInfo() {
    return axios.get("/user/user/current")
}

// 修改用户密码
export function updateAdminPassword(data) {
    return axios.post("/auth/password/update", data)
}

// 获取用户信息
export function fetchUserInfoApi(userId) {
    return axios.post("/user/user/findById", {
        id: userId,
    });
}

// 获取用户统计信息
export function fetchUserStatsApi(userId) {
    return axios.post("/count/count/user", {
        userId,
    });
}

// 获取用户笔记列表
export function fetchUserNotesApi(userId, page, size) {
    return axios.post("/note/note/UserNoteList", {
        userId,
        page,
        size,
    });
}

// 关注用户
export function followUserApi(followUserId) {
    return axios.post("/api/relation/relation/follow", {
        followUserId,
    });
}

// 取消关注用户
export function unfollowUserApi(followUserId) {
    return axios.post("/api/relation/relation/unfollow", {
        followUserId,
    });
}

// 根据用户ID查询用户信息
export function getUserInfoByIdService(params) {
    console.log('Requesting user info with params:', params);
    return axios.post("/user/user/findById", params)
        .then(response => {
            console.log('User info response:', response);
            return response;
        })
        .catch(error => {
            console.error("Error fetching user info:", error);
            console.error("Error details:", {
                status: error?.response?.status,
                data: error?.response?.data,
                config: error?.config
            });
            throw error;
        });
}