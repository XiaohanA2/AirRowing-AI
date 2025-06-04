package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;


/**
 * 评论请求对象
 */
@Data
public class CommentReqVO {

    @NotNull(message = "笔记ID不能为空")
    private Long noteId;

    private Long parentId; // 父评论ID，可为空，为空表示一级评论

    @NotBlank(message = "评论内容不能为空")
    private String content;
}
