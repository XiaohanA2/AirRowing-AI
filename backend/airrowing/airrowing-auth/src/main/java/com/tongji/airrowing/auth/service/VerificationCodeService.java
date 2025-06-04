package com.tongji.airrowing.auth.service;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.auth.model.vo.veriticationcode.SendVerificationCodeReqVO;

public interface VerificationCodeService {

    /**
     * 发送邮箱验证码
     *
     * @param sendVerificationCodeReqVO
     * @return
     */
    Response<?> send(SendVerificationCodeReqVO sendVerificationCodeReqVO);
}


