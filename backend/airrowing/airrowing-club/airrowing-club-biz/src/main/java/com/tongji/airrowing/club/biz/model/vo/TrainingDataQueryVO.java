package com.tongji.airrowing.club.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import jakarta.validation.constraints.NotNull;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class TrainingDataQueryVO {

    private Long userId; // 用户ID（这个字段是从上下文中获取的，不需要前端传递）
    private int page;    // 页码
    private int size;    // 每页大小
    private String type; // 类型（可选）
    private String startDate; // 起始日期（可选）
    private String endDate;   // 结束日期（可选）

}
