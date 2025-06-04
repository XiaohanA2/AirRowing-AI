package com.tongji.airrowing.user.biz.controller;

import com.tongji.airrowing.user.biz.domain.dataobject.UserDO;
import com.tongji.framework.biz.operationlog.aspect.ApiOperationLog;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.user.biz.model.vo.UpdateUserInfoReqVO;
import com.tongji.airrowing.user.biz.service.UserService;
import com.tongji.airrowing.user.dto.req.*;
import com.tongji.airrowing.user.dto.resp.FindUserByEmailRspDTO;
import com.tongji.airrowing.user.dto.resp.FindUserByIdRspDTO;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/user")
@Slf4j
public class UserController {

    @Resource
    private UserService userService;

    /**
     * 用户信息修改
     *
     * @param updateUserInfoReqVO
     * @return
     */
    @PostMapping(value = "/update", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Response<?> updateUserInfo(@Validated UpdateUserInfoReqVO updateUserInfoReqVO) {
        return userService.updateUserInfo(updateUserInfoReqVO);
    }

    // ===================================== 对其他服务提供的接口 =====================================
    @PostMapping("/register")
    @ApiOperationLog(description = "用户注册")
    public Response<Long> register(@Validated @RequestBody RegisterUserReqDTO registerUserReqDTO) {
        return userService.register(registerUserReqDTO);
    }

    @PostMapping("/findByEmail")
    @ApiOperationLog(description = "邮箱查询用户信息")
    public Response<FindUserByEmailRspDTO> findByEmail(@Validated @RequestBody FindUserByEmailReqDTO findUserByEmailReqDTO) {
        return userService.findByEmail(findUserByEmailReqDTO);
    }

    @PostMapping("/password/update")
    @ApiOperationLog(description = "密码更新")
    public Response<?> updatePassword(@Validated @RequestBody UpdateUserPasswordReqDTO updateUserPasswordReqDTO) {
        return userService.updatePassword(updateUserPasswordReqDTO);
    }

    @PostMapping("/findById")
    @ApiOperationLog(description = "查询用户信息")
    public Response<FindUserByIdRspDTO> findById(@Validated @RequestBody FindUserByIdReqDTO findUserByIdReqDTO) {
        return userService.findById(findUserByIdReqDTO);
    }

    @PostMapping("/findByIds")
    @ApiOperationLog(description = "批量查询用户信息")
    public Response<List<FindUserByIdRspDTO>> findByIds(@Validated @RequestBody FindUsersByIdsReqDTO findUsersByIdsReqDTO) {
        return userService.findByIds(findUsersByIdsReqDTO);
    }

    @GetMapping("/current")
    public ResponseEntity<Response<UserDO>> getCurrentUserInfo() {
        return ResponseEntity.ok(userService.getCurrentUserInfo());
    }
}


