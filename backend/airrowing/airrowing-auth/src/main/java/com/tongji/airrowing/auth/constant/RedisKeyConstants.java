package com.tongji.airrowing.auth.constant;

public class RedisKeyConstants {

    /**
     * 验证码 KEY 前缀
     */
    private static final String VERIFICATION_CODE_KEY_PREFIX = "verification_code:";


    /**
     * 构建验证码 KEY
     * @param email
     * @return
     */
    public static String buildVerificationCodeKey(String email) {
        return VERIFICATION_CODE_KEY_PREFIX + email;
    }

}
