package com.tongji.airrowing.note.biz.model.vo;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class LikedNoteVO {
    private Long id;          // 笔记ID
    private String title;     // 笔记标题
    private Long creatorId;   // 笔记创建者ID
}
