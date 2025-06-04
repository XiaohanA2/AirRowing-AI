package com.tongji.airrowing.club.biz.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Objects;

/**
 * 俱乐部成员状态枚举
 */
@Getter
@AllArgsConstructor
public enum ClubMemberStatusEnum {

    ACTIVE(0, "活跃"),
    INACTIVE(1, "非活跃"),
    REMOVED(2, "被移除");

    private final Integer code;
    private final String description;

    public static boolean isValid(Integer code) {
        for (ClubMemberStatusEnum status : ClubMemberStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return true;
            }
        }
        return false;
    }

    public static ClubMemberStatusEnum valueOf(Integer code) {
        for (ClubMemberStatusEnum status : ClubMemberStatusEnum.values()) {
            if (Objects.equals(code, status.getCode())) {
                return status;
            }
        }
        return null;
    }
}
