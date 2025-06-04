package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.Data;


/**
 * 分页查询评论请求对象
 */
@Data
public class GetCommentsReqVO {

    @NotNull(message = "笔记ID不能为空")
    private Long noteId;

    @Min(value = 1, message = "页码必须大于等于1")
    private int page;

    @Min(value = 1, message = "每页大小必须大于等于1")
    private int size;
}

