package com.tongji.airrowing.club.biz.controller;

import com.tongji.airrowing.club.biz.model.dto.*;
import com.tongji.airrowing.club.biz.model.vo.*;
import com.tongji.airrowing.club.biz.service.ClubService;
import com.tongji.framework.common.response.Response;
import com.tongji.framework.biz.operationlog.aspect.ApiOperationLog;
import jakarta.annotation.Resource;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/club")
@Slf4j
public class ClubController {

    @Resource
    private ClubService clubService;

    /**
     * 创建俱乐部
     *
     * @param createClubReqVO 请求体，包含俱乐部名称和描述
     * @return 创建俱乐部的响应数据
     */
    @PostMapping(value = "/create")
    @ApiOperationLog(description = "创建俱乐部")
    public Response<?> createClub(@Validated @RequestBody CreateClubReqVO createClubReqVO) {
        // 调用服务层创建俱乐部并返回响应
        return clubService.createClub(createClubReqVO);
    }

    @PostMapping("/update")
    public Response<?> updateClub(@Validated @RequestBody UpdateClubReqVO updateClubReqVO) {
        return clubService.updateClub(updateClubReqVO.getClubId(), updateClubReqVO.getName(), updateClubReqVO.getDescription());
    }

    @PostMapping("/delete")
    public Response<?> deleteClub(@Validated @RequestBody DeleteClubReqVO deleteClubReqVO) {
        return clubService.deleteClub(deleteClubReqVO.getClubId());
    }

    @PostMapping("/details")
    public Response<ClubDetailsDTO> getClubDetails(@Validated @RequestBody ClubDetailsReqVO clubDetailsReqVO) {
        return clubService.getClubDetails(clubDetailsReqVO.getClubId());
    }

    @PostMapping("/clublist")
    public Response<ClubListDTO> getClubList(@Validated @RequestBody ClubListReqVO clubListReqVO) {
        return clubService.getClubList(clubListReqVO.getPage(), clubListReqVO.getSize());
    }

    @PostMapping("/addHosts")
    public Response<?> addSecondHost(@RequestBody AddSecondHostReqVO addSecondHostRequestVO) {
        return clubService.addSecondHost(addSecondHostRequestVO.getClubId(), addSecondHostRequestVO.getUserId());
    }


    /**
     * 移除主持人
     *
     * @param removeHostReqVO 移除或转让请求体
     * @return 操作结果
     */
    @PostMapping("/firehost")
    public Response<?> removeHost(@Valid @RequestBody RemoveHostReqVO removeHostReqVO) {
        return clubService.removeHost(removeHostReqVO.getClubId(), removeHostReqVO.getUserId());
    }

    /**
     * 获取主持人列表
     *
     * @return 主持人列表
     */
    @PostMapping("/hostlist")
    public Response<?> getHosts(@Validated @RequestBody HostListReqVO hostListReqVO) {
        Long clubId = hostListReqVO.getClubId();
        return clubService.getHosts(clubId);
    }

    /**
     * 用户申请加入俱乐部
     * @return 操作结果
     */
    @PostMapping("/joinRequest")
    public Response<?> requestToJoinClub(@RequestBody @Valid JoinClubReqVO requestVO) {
        return clubService.requestToJoinClub(requestVO);
    }

    /**
     * 审核用户申请加入俱乐部
     *
     * @param reviewJoinReqVO 审核请求对象，包含俱乐部ID、用户ID和审核状态
     * @return 审核操作结果
     */
    @PostMapping("/reviewJoinRequest")
    public Response<?> reviewJoinRequest(@Valid @RequestBody ReviewJoinReqVO reviewJoinReqVO) {
        return clubService.reviewJoinRequest(reviewJoinReqVO);
    }

    /**
     * 用户退出俱乐部
     *
     * @return 退出操作结果
     */
    @PostMapping("/leave")
    public Response<?> leaveClub(@Valid @RequestBody LeaveClubReqVO leaveClubReqVO) {
        return clubService.leaveClub(leaveClubReqVO.getClubId());
    }

    /**
     * 获取用户的加入俱乐部申请记录列表
     *
     */
    @PostMapping("/joinRequestList")
    public Response<?> getJoinRequestList(@Valid @RequestBody JoinReqListReqVO joinRequestListReqVO) {
        int page = joinRequestListReqVO.getPage();
        int size = joinRequestListReqVO.getSize();
        return clubService.getJoinRequestList(page, size);
    }

    @PostMapping("/clubJoinRequests")
    public Response<?> getClubJoinRequests(@RequestBody @Valid ClubJoinRequestListReqVO requestVO) {
        return clubService.getClubJoinRequests(requestVO.getClubId(), requestVO.getPage(), requestVO.getSize());
    }

    /**
     * 获取俱乐部成员列表
     *
     * @return 成员列表
     */
    @PostMapping("/members")
    public Response<List<ClubMemberDTO>> getClubMembers(@RequestBody @Validated ClubMembersReqVO clubMembersReqVO) {
        Long clubId = clubMembersReqVO.getClubId();
        int page = clubMembersReqVO.getPage();
        int size = clubMembersReqVO.getSize();

        return clubService.getClubMembers(clubId, page, size);
    }

    /**
     * 发布活动
     *
     * @param publishActivityReqVO 发布活动请求对象
     * @return 创建成功的活动ID
     */
    @PostMapping("/publishActivity")
    public Response<Long> publishActivity(@Valid @RequestBody PublishActivityReqVO publishActivityReqVO) {
        return clubService.publishActivity(publishActivityReqVO);
    }

    /**
     * 更新活动
     *
     * @return 创建成功的活动ID
     */
    @PostMapping("/updateActivity")
    public Response<?> updateActivity(@RequestBody @Valid UpdateActivityReqVO updateActivityReqVO) {
        return clubService.updateActivity(updateActivityReqVO);
    }

    /**
     * 删除活动（逻辑删除）
     *
     * @param deleteActivityReqVO 删除活动请求对象
     * @return 操作结果
     */
    @PostMapping("/deleteActivity")
    public Response<?> deleteActivity(@RequestBody @Valid DeleteActivityReqVO deleteActivityReqVO) {
        return clubService.deleteActivity(deleteActivityReqVO);
    }

    @PostMapping("/activity/details")
    public Response<ActivityDetailsDTO> getActivityDetails(@RequestBody @Valid GetActivityDetailsReqVO requestVO) {
        return clubService.getActivityDetails(requestVO.getActivityId());
    }

    @PostMapping("/activityList")
    public Response<List<ActivityDetailsDTO>> getClubActivityList(@RequestBody ClubActivityListReqVO reqVO) {
        return clubService.getClubActivityList(reqVO.getClubId(), reqVO.getPage(), reqVO.getSize());
    }

    /**
     * 俱乐部成员报名参加活动
     *
     * @param signupActivityReqVO 报名活动请求对象
     * @return 报名操作结果
     */
    @PostMapping("/activity/signup")
    public Response<?> signupForActivity(@RequestBody @Valid SignupActivityReqVO signupActivityReqVO) {
        return clubService.signupForActivity(signupActivityReqVO.getActivityId());
    }

    /**
     * 审核报名申请
     *
     * @param reviewSignupReqVO 审核请求对象
     * @return 审核结果
     */
    @PostMapping("/activity/reviewSignup")
    public Response<?> reviewSignupRequest(@RequestBody @Valid ReviewSignupReqVO reviewSignupReqVO) {
        return clubService.reviewSignupRequest(reviewSignupReqVO);
    }

    /**
     * 成员取消报名
     *
     * @param cancelSignupReqVO 取消报名请求对象
     * @return 操作结果
     */
    @PostMapping("/activity/cancelSignup")
    public Response<?> cancelSignup(@RequestBody @Valid CancelSignupReqVO cancelSignupReqVO) {
        return clubService.cancelSignup(cancelSignupReqVO);
    }

    /**
     * 移除活动成员
     *
     * @param removeMemberReqVO 移除成员请求对象
     * @return 操作结果
     */
    @PostMapping("/activity/removeMember")
    public Response<?> removeMemberFromActivity(@RequestBody @Valid RemoveMemberReqVO removeMemberReqVO) {
        return clubService.removeMemberFromActivity(removeMemberReqVO);
    }

    /**
     * 移除俱乐部成员
     *
     * @param removeMemberReqVO 移除成员请求对象
     * @return 操作结果
     */
    @PostMapping("/clubRemoveMember")
    public Response<?> removeMemberFromClub(@RequestBody @Valid RemoveMemberReqVO removeMemberReqVO) {
        return clubService.removeMemberFromClub(removeMemberReqVO);
    }

    /**
     * 获取活动报名成员列表
     *
     * @param reqVO 活动报名成员列表请求对象
     * @return 报名成员列表
     */
    @PostMapping("/activity/signupMembers")
    public Response<List<ActivityMemberDTO>> getActivitySignupMembers(@RequestBody @Valid ActivitySignupMembersReqVO reqVO) {
        return clubService.getActivitySignupMembers(reqVO);
    }

    /**
     * 获取用户的活动报名申请列表
     *
     * @param reqVO 用户报名申请列表请求对象
     * @return 用户报名申请列表
     */
    @PostMapping("/activity/userSignups")
    public Response<List<ActivitySignupInfoDTO>> getUserActivitySignups(@RequestBody @Valid UserActivitySignupListReqVO reqVO) {
        return clubService.getUserActivitySignups(reqVO);
    }

    /**
     * 获取活动的报名申请列表（主持人视角）
     *
     * @param reqVO 活动报名申请列表请求对象
     * @return 报名申请列表
     */
    @PostMapping("/activity/hostSignupList")
    public Response<List<ActivitySignupDetailDTO>> getActivitySignupsForHost(@RequestBody @Valid HostActivitySignupListReqVO reqVO) {
        return clubService.getActivitySignupsForHost(reqVO);
    }



}
