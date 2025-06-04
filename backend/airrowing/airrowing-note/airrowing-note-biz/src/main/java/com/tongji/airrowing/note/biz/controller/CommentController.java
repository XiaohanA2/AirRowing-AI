package com.tongji.airrowing.note.biz.controller;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.note.biz.model.vo.*;
import com.tongji.airrowing.note.biz.service.CommentService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/comment")
@Slf4j
public class CommentController {

    @Resource
    private CommentService commentService;

    @PostMapping("/add")
    public Response<?> addComment(@Validated @RequestBody CommentReqVO commentReqVO) {
        return commentService.addComment(commentReqVO);
    }


    /**
     * 删除评论及其所有子评论
     *
     * @return 操作结果
     */
    @PostMapping("/delete")
    public Response<?> deleteComment(@Validated @RequestBody DeleteCommentReqVO deleteCommentReqVO) {
        return commentService.deleteComment(deleteCommentReqVO.getCommentId());
    }

    /**
     * 分页查询某篇笔记的一级评论
     *
     * @param request 分页请求对象
     * @return 一级评论列表（包含用户信息）
     */
    @PostMapping("/list")
    public Response<List<CommentWithUserVO>> getCommentsByNoteId(@Validated @RequestBody GetCommentsReqVO request) {
        return commentService.getCommentsByNoteId(request.getNoteId(), request.getPage(), request.getSize());
    }

    /**
     * 查询某条评论的所有子评论
     *
     * @return 子评论列表
     */
    @PostMapping("/replies")
    public Response<List<CommentRspVO>> getRepliesByCommentId(@Validated @RequestBody GetRepliesReqVO request) {
        return commentService.getRepliesByCommentId(request.getCommentId());
    }

    /**
     * 统计某篇笔记的评论总数
     *
     * @return 评论总数
     */
    @PostMapping("/count")
    public Response<Integer> countCommentsByNoteId(@RequestBody CountCommentsReqVO request) {
        return commentService.countCommentsByNoteId(request.getNoteId());
    }



}

