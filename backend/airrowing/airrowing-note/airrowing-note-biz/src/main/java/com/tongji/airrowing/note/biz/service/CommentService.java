package com.tongji.airrowing.note.biz.service;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.note.biz.model.vo.CommentReqVO;
import com.tongji.airrowing.note.biz.model.vo.CommentRspVO;
import com.tongji.airrowing.note.biz.model.vo.CommentWithUserVO;

import java.util.List;


/**
 * 评论服务接口
 */
public interface CommentService {

    /**
     * 添加评论
     *
     * @param commentReqVO 评论请求对象
     * @return 操作结果
     */
    Response<?> addComment(CommentReqVO commentReqVO);

    /**
     * 删除评论及其所有子评论
     *
     * @param commentId 评论ID
     * @return 操作结果
     */
    Response<?> deleteComment(Long commentId);

    /**
     * 分页查询某篇笔记的一级评论
     *
     * @param noteId 笔记ID
     * @param page   页码
     * @param size   每页大小
     * @return 一级评论列表（包含用户信息）
     */
    Response<List<CommentWithUserVO>> getCommentsByNoteId(Long noteId, int page, int size);

    /**
     * 查询某条评论的所有子评论
     *
     * @param commentId 父评论ID
     * @return 子评论列表
     */
    Response<List<CommentRspVO>> getRepliesByCommentId(Long commentId);

    /**
     * 统计某篇笔记的评论总数
     *
     * @param noteId 笔记ID
     * @return 评论总数
     */
    Response<Integer> countCommentsByNoteId(Long noteId);
}
