package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;


/**
 * 获取子评论请求对象
 */
@Data
public class GetRepliesReqVO {
    @NotNull(message = "评论ID不能为空")
    private Long commentId;
}
