package com.tongji.airrowing.club.biz.model.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ClubMemberDTO {
    private Long userId;
    private String nickname;
    private String avatar;
    private LocalDateTime joinTime;
}
