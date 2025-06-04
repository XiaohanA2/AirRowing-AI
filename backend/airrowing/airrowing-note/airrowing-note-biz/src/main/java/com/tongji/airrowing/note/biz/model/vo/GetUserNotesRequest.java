package com.tongji.airrowing.note.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class GetUserNotesRequest {
    private Long userId;  // 当前用户的ID
    private int page;     // 当前页码
    private int size;     // 每页数量
    private Integer isTop;  // 是否置顶（可选）
    private Integer visible;  // 可见性（可选）
    private Integer status;  // 状态（可选）
}
