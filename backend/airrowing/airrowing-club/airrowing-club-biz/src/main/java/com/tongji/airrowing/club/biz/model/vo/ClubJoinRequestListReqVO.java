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
public class ClubJoinRequestListReqVO {

    @NotNull(message = "俱乐部ID不能为空")
    private Long clubId;

    @Min(value = 1, message = "页码必须大于0")
    private int page;

    @Min(value = 1, message = "每页记录数必须大于0")
    private int size;
}

