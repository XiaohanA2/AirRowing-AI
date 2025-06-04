package com.tongji.airrowing.club.biz.enums;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Objects;

/**
 * 主持人级别枚举
 */
@Getter
@AllArgsConstructor
public enum ClubHostLevelEnum {

    FIRST(1, "第一主持人"),
    SECOND(2, "第二主持人");

    private final Integer code;
    private final String description;

    /**
     * 判断级别是否有效
     *
     * @param code
     * @return
     */
    public static boolean isValid(Integer code) {
        for (ClubHostLevelEnum level : ClubHostLevelEnum.values()) {
            if (Objects.equals(code, level.getCode())) {
                return true;
            }
        }
        return false;
    }

    /**
     * 根据级别 code 获取对应的枚举
     *
     * @param code
     * @return
     */
    public static ClubHostLevelEnum valueOf(Integer code) {
        for (ClubHostLevelEnum level : ClubHostLevelEnum.values()) {
            if (Objects.equals(code, level.getCode())) {
                return level;
            }
        }
        return null;
    }
}

