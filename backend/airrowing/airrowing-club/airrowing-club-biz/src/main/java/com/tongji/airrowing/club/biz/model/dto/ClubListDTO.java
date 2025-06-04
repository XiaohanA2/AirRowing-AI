package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class ClubListDTO {

    private List<ClubItemDTO> clubs;
    private int currentPage;
    private int pageSize;
    private long total;

    @Data
    public static class ClubItemDTO {
        private Long clubId;
        private String name;
        private String description;
        private LocalDateTime createTime;
    }
}

