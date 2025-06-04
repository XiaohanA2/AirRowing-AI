package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;


/**
 * 请求对象：统计某篇笔记的评论总数
 */
@Data
public class CountCommentsReqVO {

    @NotNull(message = "笔记ID不能为空")
    private Long noteId;
}
