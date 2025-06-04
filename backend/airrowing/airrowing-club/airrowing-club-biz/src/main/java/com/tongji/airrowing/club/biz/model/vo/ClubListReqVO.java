package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.Min;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ClubListReqVO {

    @Min(value = 1, message = "页码必须大于等于1")
    private int page = 1;

    @Min(value = 1, message = "每页数量必须大于等于1")
    private int size = 10;
}
