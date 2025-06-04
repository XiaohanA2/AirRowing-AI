package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class DeleteCommentReqVO {

    @NotNull(message = "评论ID不能为空")
    private Long commentId;
}
