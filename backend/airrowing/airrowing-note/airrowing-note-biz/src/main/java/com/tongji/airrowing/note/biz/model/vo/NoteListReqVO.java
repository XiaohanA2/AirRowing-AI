package com.tongji.airrowing.note.biz.model.vo;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class NoteListReqVO {

    private int page;   // 页码，表示请求哪一页
    private int size;   // 每页显示多少条记录
}
