package com.tongji.airrowing.user.biz.service;

import com.tongji.airrowing.user.biz.domain.dataobject.UserDO;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.user.biz.model.vo.UpdateUserInfoReqVO;
import com.tongji.airrowing.user.dto.req.*;
import com.tongji.airrowing.user.dto.resp.FindUserByEmailRspDTO;
import com.tongji.airrowing.user.dto.resp.FindUserByIdRspDTO;
import com.tongji.airrowing.user.dto.resp.FindUserByNicknameRspDTO;

import java.util.List;


public interface UserService {

    /**
     * 更新用户信息
     *
     * @param updateUserInfoReqVO
     * @return
     */
    Response<?> updateUserInfo(UpdateUserInfoReqVO updateUserInfoReqVO);

    /**
     * 用户注册
     *
     * @param registerUserReqDTO
     * @return
     */
    Response<Long> register(RegisterUserReqDTO registerUserReqDTO);

    /**
     * 根据邮箱查询用户信息
     *
     * @param findUserByEmailReqDTO
     * @return
     */
    Response<FindUserByEmailRspDTO> findByEmail(FindUserByEmailReqDTO findUserByEmailReqDTO);

    //根据昵称查找
    Response<FindUserByNicknameRspDTO> findByNickname(FindUserByNicknameReqDTO findUserByNicknameReqDTO);

    /**
     * 更新密码
     *
     * @param updateUserPasswordReqDTO
     * @return
     */
    Response<?> updatePassword(UpdateUserPasswordReqDTO updateUserPasswordReqDTO);

    /**
     * 根据用户 ID 查询用户信息
     *
     * @param findUserByIdReqDTO
     * @return
     */
    Response<FindUserByIdRspDTO> findById(FindUserByIdReqDTO findUserByIdReqDTO);

    /**
     * 批量根据用户 ID 查询用户信息
     *
     * @param findUsersByIdsReqDTO
     * @return
     */
    Response<List<FindUserByIdRspDTO>> findByIds(FindUsersByIdsReqDTO findUsersByIdsReqDTO);

    Response<UserDO> getCurrentUserInfo();
}

