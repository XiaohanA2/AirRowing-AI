package com.tongji.airrowing.club.biz.model.dto;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class UpdatedClubResponseDTO {

    private Long clubId;
    private String name;
    private String description;
    private LocalDateTime updateTime;

    public UpdatedClubResponseDTO(Long clubId, String name, String description, LocalDateTime updateTime) {
        this.clubId = clubId;
        this.name = name;
        this.description = description;
        this.updateTime = updateTime;
    }
}

