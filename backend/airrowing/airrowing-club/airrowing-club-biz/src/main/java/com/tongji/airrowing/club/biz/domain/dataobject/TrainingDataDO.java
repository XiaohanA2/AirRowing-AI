package com.tongji.airrowing.club.biz.domain.dataobject;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class TrainingDataDO {
    private Long id;

    private Long userId;

    private LocalDateTime trainingDate;

    private Integer duration;

    private BigDecimal distance;

    private BigDecimal calories;

    private Byte status;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;

    private String type;
}