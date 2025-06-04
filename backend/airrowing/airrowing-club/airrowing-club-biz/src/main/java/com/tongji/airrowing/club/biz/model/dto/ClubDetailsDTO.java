package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class ClubDetailsDTO {
    private Long clubId;
    private String name;
    private String description;
    private Long creatorId;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public ClubDetailsDTO(Long clubId, String name, String description, Long creatorId, LocalDateTime createTime, LocalDateTime updateTime) {
        this.clubId = clubId;
        this.name = name;
        this.description = description;
        this.creatorId = creatorId;
        this.createTime = createTime;
        this.updateTime = updateTime;
    }
}
