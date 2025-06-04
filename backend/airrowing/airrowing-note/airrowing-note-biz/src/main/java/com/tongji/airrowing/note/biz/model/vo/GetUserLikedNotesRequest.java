package com.tongji.airrowing.note.biz.model.vo;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 分页查询用户点赞笔记的请求参数
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class GetUserLikedNotesRequest {

    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @Min(value = 1, message = "页码必须大于等于1")
    private int page;

    @Min(value = 1, message = "每页大小必须大于等于1")
    private int size;
}
