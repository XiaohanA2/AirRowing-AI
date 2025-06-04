package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class JoinReqListReqVO {

    /**
     * 页码，从1开始
     */
    @Min(value = 1, message = "页码必须大于0")
    private int page;

    /**
     * 每页记录数
     */
    @Min(value = 1, message = "每页记录数必须大于0")
    private int size;
}
