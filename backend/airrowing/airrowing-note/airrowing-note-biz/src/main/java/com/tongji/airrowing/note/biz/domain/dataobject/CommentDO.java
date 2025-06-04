package com.tongji.airrowing.note.biz.domain.dataobject;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class CommentDO {
    private Long id;

    private Long noteId;

    private Long userId;

    private Long parentId;

    private Byte status;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;

    private String content;
}