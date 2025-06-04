package com.tongji.airrowing.club.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class TrainingDataStatisticsRequest {

    private String type;  // 训练类型，如：rowing, cycling 等
    private String startDate;  // 开始日期（格式：yyyy-MM-dd）
    private String endDate;    // 结束日期（格式：yyyy-MM-dd）

}
