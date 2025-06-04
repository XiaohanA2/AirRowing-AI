package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class HostActivitySignupListReqVO {
    @NotNull(message = "活动ID不能为空")
    private Long activityId;

    @NotNull(message = "页码不能为空")
    private Integer page;

    @NotNull(message = "每页记录数不能为空")
    private Integer size;
}
