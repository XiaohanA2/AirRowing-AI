package com.tongji.airrowing.club.biz.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PastOrPresent;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 用户提交的训练数据请求体
 */
@Data
public class TrainingDataVO {

    @NotNull(message = "训练日期不能为空")
    @PastOrPresent(message = "训练日期必须是当前时间或之前")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime trainingDate;

    @NotNull(message = "训练时长不能为空")
    @Positive(message = "训练时长必须是正数")
    private Integer duration; // 训练时长（单位：分钟）

    @NotNull(message = "训练距离不能为空")
    @Positive(message = "训练距离必须是正数")
    private BigDecimal distance; // 训练距离（单位：公里）

    private BigDecimal calories; // 消耗的卡路里（单位：大卡）

    private Byte status; // 训练状态（例如，1表示完成，0表示未完成）

    @NotNull(message = "训练类型不能为空")
    private String type; // 训练类型
}
