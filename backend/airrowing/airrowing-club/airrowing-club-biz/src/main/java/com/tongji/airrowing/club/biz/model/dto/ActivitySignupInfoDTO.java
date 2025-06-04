package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ActivitySignupInfoDTO {
    private Long activityId;
    private String activityTitle;
    private LocalDateTime signupTime;
    private Integer status;
}

