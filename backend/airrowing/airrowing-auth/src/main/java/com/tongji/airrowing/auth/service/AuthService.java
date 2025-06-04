package com.tongji.airrowing.auth.service;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.auth.model.vo.user.UpdatePasswordReqVO;
import com.tongji.airrowing.auth.model.vo.user.UserLoginReqVO;

public interface AuthService {

    /**
     * 登录与注册
     * @param userLoginReqVO
     * @return
     */
    Response<String> loginAndRegister(UserLoginReqVO userLoginReqVO);

    /**
     * 退出登录
     * @return
     */
    Response<?> logout();

    /**
     * 修改密码
     * @param updatePasswordReqVO
     * @return
     */
    Response<?> updatePassword(UpdatePasswordReqVO updatePasswordReqVO);

}


