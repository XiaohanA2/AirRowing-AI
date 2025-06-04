package com.tongji.airrowing.club.biz.domain.dataobject;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder

public class ActivitySignupDO {
    private Long id;

    private Long activityId;

    private Long userId;

    private LocalDateTime signupTime;

    private Integer status;

    private LocalDateTime updateTime;
}