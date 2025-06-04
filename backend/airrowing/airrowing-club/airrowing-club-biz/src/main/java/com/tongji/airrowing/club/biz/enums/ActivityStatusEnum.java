package com.tongji.airrowing.club.biz.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Objects;

/**
 * 活动状态枚举
 */
@Getter
@AllArgsConstructor
public enum ActivityStatusEnum {

    PENDING(0, "待开始"),
    ONGOING(1, "进行中"),
    COMPLETED(2, "已完成"),
    CANCELED(3, "已取消");

    private final Integer code;
    private final String description;

    public static boolean isValid(Integer code) {
        for (ActivityStatusEnum status : ActivityStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return true;
            }
        }
        return false;
    }

    public static ActivityStatusEnum valueOf(Integer code) {
        for (ActivityStatusEnum status : ActivityStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return status;
            }
        }
        return null;
    }
}
