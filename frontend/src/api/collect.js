import axios from "@/axios";
import { getToken } from "@/composables/cookie"

//获取用户收藏笔记信息
export function collectListService(params) {
    return axios.post("/note/note/collected/list", params)
}

//获取用户点赞笔记信息
export function likedListService(params) {
    return axios.post("/note/note/liked/list", params)
}

//查看收藏笔记详情
export function collectListDetailService(params) {
    const token = getToken();
    return axios.post("/note/note/detail", params, {
        headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
        }
    });
}

//笔记收藏
export function doCollectService(params) {
    return axios.post("/note/note/collect", params)
}

//笔记取消收藏
export function unCollectService(params) {
    return axios.post("/note/note/uncollect", params)
}

//点赞列表
export function likeListService(params) {
    return axios.post("/note/note/liked/list", params)
}

//点赞笔记
export function dolikeNoteService(params) {
    return axios.post("/note/note/like", params)
}

//取消点赞笔记
export function nolikeNoteService(params) {
    return axios.post("/note/note/unlike", params)
}

//获取笔记计数信息
// export function noteCountService(params) {
//     const token = getToken();
//     return axios.post("/count/count/note", params, {
//         headers: {
//             Authorization: `Bearer ${token}`,
//             'Content-Type': 'application/json'
//         }
//     });
// }
//获取笔记计数信息
export function noteCountService(params) {
    return axios.post("/count/count/note", params, { silent: true })
}

//获取笔记一级评论
export function noteFirstCommentService(params) {
    return axios.post("/note/comment/list", params)
}

//获取笔记评论数
export function noteCommentCountService(params) {
    return axios.post("/note/comment/count", params)
}

//删除评论
export function deleteCommentService(params) {
    return axios.post("/note/comment/delete", params)
}

//获取评论的评论
export function subCommentService(params) {
    return axios.post("/note/comment/replies", params)
}

//增加评论
export function addCommentService(params) {
    return axios.post("/note/comment/add", params)
}