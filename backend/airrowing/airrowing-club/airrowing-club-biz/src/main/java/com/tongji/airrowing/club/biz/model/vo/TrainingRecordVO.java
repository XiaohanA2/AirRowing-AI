package com.tongji.airrowing.club.biz.model.vo;

import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class TrainingRecordVO {

    private Long id;
    private LocalDateTime trainingDate;
    private Integer duration;
    private BigDecimal distance;
    private BigDecimal calories;
    private String type;

    // 构造函数
    public TrainingRecordVO(Long id, LocalDateTime trainingDate, Integer duration,
                            BigDecimal distance, BigDecimal calories, String type) {
        this.id = id;
        this.trainingDate = trainingDate;
        this.duration = duration;
        this.distance = distance;
        this.calories = calories;
        this.type = type;
    }
}
