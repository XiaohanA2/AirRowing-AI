package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class ActivitySignupMembersReqVO {

    @NotNull(message = "活动ID不能为空")
    private Long activityId;

    private Integer page = 1;
    private Integer size = 10;
}
