package com.tongji.airrowing.club.biz.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Objects;

/**
 * 俱乐部逻辑删除状态枚举
 */
@Getter
@AllArgsConstructor
public enum ClubDeletionStatusEnum {

    NOT_DELETED(0, "未删除"),
    DELETED(1, "已删除");

    private final Integer code;
    private final String description;

    public static boolean isValid(Integer code) {
        for (ClubDeletionStatusEnum status : ClubDeletionStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return true;
            }
        }
        return false;
    }

    public static ClubDeletionStatusEnum valueOf(Integer code) {
        for (ClubDeletionStatusEnum status : ClubDeletionStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return status;
            }
        }
        return null;
    }
}
