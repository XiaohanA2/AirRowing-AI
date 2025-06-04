import axios from "@/axios";
import { ref } from 'vue'

//获取当前用户信息
export function currentUserService() {
    return axios.get("/user/user/current")
}
//获取关注列表
export function followService(params) {
    return axios.post("/relation/relation/following/list", params)
}

//获取粉丝列表
export function fansService(params) {
    return axios.post("/relation/relation/fans/list", params)
}

//取关
export function noFollowService(params) {
    return axios.post("/relation/relation/unfollow", params)
}
//关注
export function doFollowService(params) {
    return axios.post("/relation/relation/follow", params)
}