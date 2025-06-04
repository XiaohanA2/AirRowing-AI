package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ReviewJoinReqVO {

    /**
     * 俱乐部ID
     */
    @NotNull(message = "俱乐部ID不能为空")
    private Long clubId;

    /**
     * 用户ID
     */
    @NotNull(message = "用户ID不能为空")
    private Long userId;

    /**
     * 审核状态 (1: 同意, 2: 拒绝)
     */
    @NotNull(message = "审核状态不能为空")
    private Integer status;
}
