package com.tongji.airrowing.club.biz.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class JoinRequestDTO {
    private Long clubId;
    private Long userId;
    private int status;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
