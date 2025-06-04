import axios from "@/axios";

//获取用户收藏笔记信息
export function collectListService(params) {
    return axios.post("/note/note/collected/list", params)
}

//获取用户点赞笔记信息
export function likedListService(params) {
    return axios.post("/note/note/liked/list", params)
}

// 获取所有笔记列表
export function getNoteListService(params) {
    return axios.post("/note/note/list", params)
        .then(response => response.data)
        .catch(error => {
            console.error("Error fetching note list:", error);
            throw error;
        });
}

// 查看笔记详情
export function getNoteDetailService(params) {
    return axios.post("/note/note/detail", params);
}

// 根据用户ID查询用户信息
export function getUserInfoByIdService(params) {
    return axios.post("/user/user/findById", params);
}

// 增加评论
export function addCommentService(params) {
    return axios.post('/note/comment/add', params);
}

// 查看笔记的一级评论
export function getNoteCommentsService(params) {
    return axios.post('/note/comment/list', params);
}

// 查看评论的回复
export function getCommentRepliesService(params) {
    return axios.post('/note/comment/replies', params);
}

// 删除评论及其子评论
export function deleteCommentService(params) {
    return axios.post('/note/comment/delete', params);
}

// 添加获取评论总数的接口
export function getCommentCountService(params) {
    return axios.post("/note/comment/count", params);
}