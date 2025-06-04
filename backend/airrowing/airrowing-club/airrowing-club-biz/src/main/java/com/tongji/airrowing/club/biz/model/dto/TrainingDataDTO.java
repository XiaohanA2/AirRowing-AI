package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class TrainingDataDTO {

    private Long userId;

    private LocalDateTime trainingDate;

    private Integer duration; // 训练时长（单位：分钟）

    private BigDecimal distance; // 训练距离（单位：公里）

    private BigDecimal calories; // 消耗的卡路里（单位：大卡）

    private Byte status; // 训练状态（例如，1表示完成，0表示未完成）

    private String type;
}

