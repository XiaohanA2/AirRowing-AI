package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ReviewSignupReqVO {

    @NotNull(message = "活动ID不能为空")
    private Long activityId;

    @NotNull(message = "用户ID不能为空")
    private Long userId;

    @NotNull(message = "审核状态不能为空") // 1表示通过，2表示拒绝
    private Integer status;
}
