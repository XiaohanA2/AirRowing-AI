package com.tongji.airrowing.club.biz.domain.dataobject;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ClubJoinReqDO {
    private Long id;

    private Long clubId;

    private Long userId;

    private Integer status;

    private LocalDateTime createTime;

    private LocalDateTime updateTime;

}