package com.tongji.airrowing.club.biz.model.vo;

import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class RemoveHostReqVO {

    @NotNull(message = "俱乐部ID不能为空")
    private Long clubId;

    @NotNull(message = "用户ID不能为空")
    private Long userId;
}
