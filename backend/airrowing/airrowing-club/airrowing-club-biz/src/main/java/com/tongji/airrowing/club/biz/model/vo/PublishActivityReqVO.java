package com.tongji.airrowing.club.biz.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PublishActivityReqVO {

    /**
     * 俱乐部ID
     */
    @NotNull(message = "俱乐部ID不能为空")
    private Long clubId;

    /**
     * 活动标题
     */
    @NotNull(message = "活动标题不能为空")
    private String title;

    /**
     * 活动内容
     */
    @NotNull(message = "活动内容不能为空")
    private String content;

    /**
     * 活动开始时间
     */
    @NotNull(message = "活动开始时间不能为空")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime startTime;

    /**
     * 活动结束时间
     */
    @NotNull(message = "活动结束时间不能为空")
    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime endTime;
}
