package com.tongji.airrowing.note.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 用户收藏的笔记视图对象
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CollectedNoteVO {
    private Long id;          // 笔记ID
    private String title;     // 笔记标题
    private Long creatorId;   // 笔记创建者ID
}