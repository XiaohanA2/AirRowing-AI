package com.tongji.airrowing.note.biz.model.vo;

import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 评论响应对象
 */
@Data
public class CommentRspVO {

    private Long id;               // 评论ID
    private Long noteId;           // 笔记ID
    private Long userId;           // 用户ID
    private String userName;       // 用户昵称
    private String userAvatar;     // 用户头像
    private String content;        // 评论内容
    private Long parentId;         // 父评论ID
    private LocalDateTime createTime; // 创建时间
    private LocalDateTime updateTime; // 更新时间
    private List<CommentRspVO> replies; // 子评论列表
}
