package com.tongji.airrowing.club.biz.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Objects;

/**
 * 活动报名状态枚举
 */
@Getter
@AllArgsConstructor
public enum SignupStatusEnum {

    REGISTERED(1, "已报名"),
    CANCELED(0, "已取消");

    private final Integer code;
    private final String description;

    public static boolean isValid(Integer code) {
        for (SignupStatusEnum status : SignupStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return true;
            }
        }
        return false;
    }

    public static SignupStatusEnum valueOf(Integer code) {
        for (SignupStatusEnum status : SignupStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return status;
            }
        }
        return null;
    }
}

