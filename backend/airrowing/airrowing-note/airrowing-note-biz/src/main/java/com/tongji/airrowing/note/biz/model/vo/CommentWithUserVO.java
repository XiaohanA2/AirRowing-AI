package com.tongji.airrowing.note.biz.model.vo;

import lombok.Data;

import java.time.LocalDateTime;

/**
 * 评论与用户信息的响应对象
 */
@Data
public class CommentWithUserVO {

    private Long commentId;         // 评论ID
    private Long noteId;            // 笔记ID
    private Long userId;            // 用户ID
    private String userName;        // 用户昵称
    private String userAvatar;      // 用户头像
    private String content;         // 评论内容
    private Long parentId;          // 父评论ID
    private LocalDateTime createTime; // 创建时间
    private LocalDateTime updateTime; // 更新时间
}

