package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ActivitySignupDetailDTO {
    private Long userId;
    private String username;
    private String profilePicture;
    private LocalDateTime signupTime;
    private Integer status;
}

