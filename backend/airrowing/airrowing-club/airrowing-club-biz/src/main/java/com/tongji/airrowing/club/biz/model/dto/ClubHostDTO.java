package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ClubHostDTO {
    private Long userId;
    private Integer hostLevel;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public ClubHostDTO(Long userId, Integer hostLevel, LocalDateTime createTime, LocalDateTime updateTime) {
        this.userId = userId;
        this.hostLevel = hostLevel;
        this.createTime = createTime;
        this.updateTime = updateTime;
    }
}
