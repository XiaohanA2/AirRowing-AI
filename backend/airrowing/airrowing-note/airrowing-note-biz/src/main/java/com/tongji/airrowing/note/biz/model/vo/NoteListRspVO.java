package com.tongji.airrowing.note.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class NoteListRspVO {
    private Long id;
    private String title;
    private Long creatorId;  // 发布者ID
}
