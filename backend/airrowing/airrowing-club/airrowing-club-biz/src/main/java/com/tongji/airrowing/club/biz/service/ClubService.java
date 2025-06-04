package com.tongji.airrowing.club.biz.service;

import com.tongji.airrowing.club.biz.domain.dataobject.ClubDO;
import com.tongji.airrowing.club.biz.domain.dataobject.ClubJoinReqDO;
import com.tongji.airrowing.club.biz.model.dto.*;
import com.tongji.airrowing.club.biz.model.vo.*;
import com.tongji.framework.common.response.Response;

import java.util.List;

public interface ClubService {
    /**
     * 创建俱乐部
     * @param createClubReqVO 包含俱乐部创建信息的请求对象
     * @return 创建成功后的俱乐部 ID
     */
    Response<Long> createClub(CreateClubReqVO createClubReqVO);

    /**
     * 更新俱乐部信息
     *
     * @param clubId 俱乐部 ID
     * @param name 新的俱乐部名称
     * @param description 新的俱乐部描述
     * @return 更新后的俱乐部信息
     */
    Response<UpdatedClubResponseDTO> updateClub(Long clubId, String name, String description);

    /**
     * 删除俱乐部（逻辑删除）
     *
     * @param clubId 俱乐部 ID
     * @return 删除操作的响应
     */
    Response<?> deleteClub(Long clubId);

    /**
     * 获取俱乐部详情
     *
     * @param clubId 俱乐部 ID
     * @return 俱乐部详情
     */
    Response<ClubDetailsDTO> getClubDetails(Long clubId);


    /**
     * 获取俱乐部列表（分页）
     *
     * @param page 页码
     * @param size 每页数量
     * @return 俱乐部列表（分页）
     */
    Response<ClubListDTO> getClubList(int page, int size);

    /**
     * 添加第二主持人
     *
     * @param clubId 俱乐部 ID
     * @param userId 用户 ID（将成为第二主持人）
     * @return 操作结果
     */
    Response<?> addSecondHost(Long clubId, Long userId);

    /**
     * 移除或转让主持人身份
     *
     * @param clubId 俱乐部ID
     * @param userId 要移除或转让的用户ID
     * @return 操作结果
     */
    Response<?> removeHost(Long clubId, Long userId);

    /**
     * 获取指定俱乐部的主持人列表
     *
     * @param clubId 俱乐部 ID
     * @return 主持人列表
     */
    Response<List<ClubHostDTO>> getHosts(Long clubId);

    /**
     * 用户申请加入俱乐部
     * @param requestVO 包含 clubId 的 VO 对象
     * @return 操作结果
     */
    Response<?> requestToJoinClub(JoinClubReqVO requestVO);

    /**
     * 审核用户申请加入俱乐部
     *
     * @param reviewJoinReqVO 审核请求对象，包含俱乐部ID、用户ID和审核状态
     * @return 操作结果
     */
    Response<?> reviewJoinRequest(ReviewJoinReqVO reviewJoinReqVO);

    /**
     * 用户退出俱乐部
     *
     * @param clubId 俱乐部ID
     * @return 退出操作结果
     */
    Response<?> leaveClub(Long clubId);

    /**
     * 获取用户申请加入俱乐部的记录列表，支持分页
     *
     * @param page 页码
     * @param size 每页记录数
     * @return 分页的加入申请记录列表
     */
    Response<List<JoinRequestDTO>> getJoinRequestList(int page, int size);

    Response<List<JoinRequestDTO>> getClubJoinRequests(Long clubId, int page, int size);

    // 获取俱乐部成员列表接口
    Response<List<ClubMemberDTO>> getClubMembers(Long clubId, int page, int size);

    // 发布活动
    Response<Long> publishActivity(PublishActivityReqVO publishActivityReqVO);

    /**
     * 更新活动信息
     *
     * @param updateActivityReqVO 更新活动请求对象
     * @return 操作结果
     */
    Response<?> updateActivity(UpdateActivityReqVO updateActivityReqVO);

    /**
     * 删除活动（逻辑删除）
     *
     * @param deleteActivityReqVO 删除活动请求对象
     * @return 操作结果
     */
    Response<?> deleteActivity(DeleteActivityReqVO deleteActivityReqVO);

    Response<ActivityDetailsDTO> getActivityDetails(Long activityId);

    /**
     * 获取指定俱乐部的活动列表
     *
     * @param clubId 俱乐部ID
     * @param page 页码
     * @param size 每页大小
     * @return 活动详情DTO列表的响应
     */
    Response<List<ActivityDetailsDTO>> getClubActivityList(Long clubId, int page, int size);

    //活动报名
    Response<?> signupForActivity(Long activityId);

    //主持人审核活动报名
    Response<?> reviewSignupRequest(ReviewSignupReqVO reviewSignupReqVO);

    //成员取消报名
    Response<?> cancelSignup(CancelSignupReqVO cancelSignupReqVO);

    Response<?> removeMemberFromActivity(RemoveMemberReqVO removeMemberReqVO);

    Response<?> removeMemberFromClub(RemoveMemberReqVO removeMemberReqVO);

    Response<List<ActivityMemberDTO>> getActivitySignupMembers(ActivitySignupMembersReqVO reqVO);

    Response<List<ActivitySignupInfoDTO>> getUserActivitySignups(UserActivitySignupListReqVO reqVO);

    Response<List<ActivitySignupDetailDTO>> getActivitySignupsForHost(HostActivitySignupListReqVO reqVO);

}

