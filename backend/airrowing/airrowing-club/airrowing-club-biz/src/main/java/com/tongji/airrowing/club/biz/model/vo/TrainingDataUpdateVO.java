package com.tongji.airrowing.club.biz.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.math.BigDecimal;

@Data
public class TrainingDataUpdateVO {
    @NotNull(message = "训练ID不能为空")
    private Long id;  // 训练记录 ID（必填）
    private Integer duration;  // 训练时长（可选）
    private BigDecimal distance;  // 训练距离（可选）
    private BigDecimal calories;  // 消耗的卡路里（可选）
    private String type;  // 训练类型（可选）

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private String trainingDate;  // 训练日期（可选）
}
