package com.tongji.airrowing.user.api;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.user.constant.ApiConstants;
import com.tongji.airrowing.user.dto.req.*;
import com.tongji.airrowing.user.dto.resp.FindUserByEmailRspDTO;
import com.tongji.airrowing.user.dto.resp.FindUserByIdRspDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.List;

@FeignClient(name = ApiConstants.SERVICE_NAME)
public interface UserFeignApi {

    String PREFIX = "/user";

    /**
     * 用户注册
     *
     * @param registerUserReqDTO
     * @return
     */
    @PostMapping(value = PREFIX + "/register")
    Response<Long> registerUser(@RequestBody RegisterUserReqDTO registerUserReqDTO);

    /**
     * 根据邮箱查询用户信息
     *
     * @param findUserByEmailReqDTO
     * @return
     */
    @PostMapping(value = PREFIX + "/findByEmail")
    Response<FindUserByEmailRspDTO> findByEmail(@RequestBody FindUserByEmailReqDTO findUserByEmailReqDTO);

    /**
     * 更新密码
     *
     * @param updateUserPasswordReqDTO
     * @return
     */
    @PostMapping(value = PREFIX + "/password/update")
    Response<?> updatePassword(@RequestBody UpdateUserPasswordReqDTO updateUserPasswordReqDTO);

    /**
     * 根据用户 ID 查询用户信息
     *
     * @param findUserByIdReqDTO
     * @return
     */
    @PostMapping(value = PREFIX + "/findById")
    Response<FindUserByIdRspDTO> findById(@RequestBody FindUserByIdReqDTO findUserByIdReqDTO);

    /**
     * 批量查询用户信息
     *
     * @param findUsersByIdsReqDTO
     * @return
     */
    @PostMapping(value = PREFIX + "/findByIds")
    Response<List<FindUserByIdRspDTO>> findByIds(@RequestBody FindUsersByIdsReqDTO findUsersByIdsReqDTO);
}
