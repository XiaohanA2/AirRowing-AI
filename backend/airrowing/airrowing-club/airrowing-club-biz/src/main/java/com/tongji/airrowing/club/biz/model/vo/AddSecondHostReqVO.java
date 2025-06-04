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
public class AddSecondHostReqVO {
    @NotNull(message = "俱乐部 ID 不能为空")
    private Long clubId;

    @NotNull(message = "用户 ID 不能为空")
    private Long userId;
}
