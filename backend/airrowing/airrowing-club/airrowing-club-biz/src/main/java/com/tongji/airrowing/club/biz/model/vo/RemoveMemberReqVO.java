package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class RemoveMemberReqVO {

    @NotNull(message = "俱乐部ID不能为空")
    private Long clubId;

    @NotNull(message = "用户ID不能为空")
    private Long userId;

    private Long activityId; // 可选，指定活动ID时表示移除活动成员
}

