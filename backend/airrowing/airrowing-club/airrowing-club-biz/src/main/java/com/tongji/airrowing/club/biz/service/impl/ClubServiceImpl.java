package com.tongji.airrowing.club.biz.service.impl;

import com.tongji.airrowing.club.biz.domain.dataobject.*;
import com.tongji.airrowing.club.biz.domain.mapper.*;
import com.tongji.airrowing.club.biz.model.dto.*;
import com.tongji.airrowing.club.biz.model.vo.*;
import com.tongji.airrowing.club.biz.service.ClubService;
import com.tongji.framework.common.exception.BizException;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.club.biz.enums.ResponseCodeEnum;
import com.tongji.airrowing.club.biz.rpc.DistributedIdGeneratorRpcService;
import com.tongji.framework.biz.context.holder.LoginUserContextHolder;
import jakarta.annotation.Resource;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
@Slf4j
public class ClubServiceImpl implements ClubService {

    @Resource
    private UserDOMapper userDOMapper;

    @Resource
    private ClubDOMapper clubDOMapper;

    @Resource
    private ActivityDOMapper activityDOMapper;

    @Resource
    private ClubHostDOMapper clubHostMapper;

    @Resource
    private ClubMemberDOMapper clubMemberDOMapper;

    @Resource
    private ClubJoinReqDOMapper clubJoinReqMapper;

    @Resource
    private ActivitySignupDOMapper activitySignupDOMapper;

    @Resource
    private ActivityMemberDOMapper activityMemberDOMapper;

    @Resource
    private DistributedIdGeneratorRpcService distributedIdGeneratorRpcService;

    /**
     * 创建俱乐部
     *
     * @param createClubReqVO 创建俱乐部的请求对象
     * @return 创建成功后的响应
     */
    @Override
    @Transactional
    public Response<Long> createClub(CreateClubReqVO createClubReqVO) {

        // 校验俱乐部名称是否为空
        if (StringUtils.isBlank(createClubReqVO.getName())) {
            throw new BizException(ResponseCodeEnum.CLUB_NAME_EMPTY); // 如果俱乐部名称为空，则抛出异常
        }

        // 获取当前登录用户的 ID 作为创建者 ID
        Long creatorId = LoginUserContextHolder.getUserId();
        if (creatorId == null) {
            throw new BizException(ResponseCodeEnum.ID_NOT_VALID);
        }

        // 调用分布式 ID 生成服务，生成俱乐部 ID
        String clubId = distributedIdGeneratorRpcService.getSnowflakeId();

        // 创建 ClubDO 对象并设置值
        ClubDO clubDO = ClubDO.builder()
                .id(Long.valueOf(clubId))
                .name(createClubReqVO.getName())
                .description(createClubReqVO.getDescription())
                .creatorId(creatorId) // 使用从上下文中获取的创建者 ID
                .createTime(LocalDateTime.now())
                .updateTime(LocalDateTime.now())
                .isDeleted(false) // 新创建的俱乐部默认未删除
                .build();

        try {
            // 将俱乐部插入到 t_club 表
            clubDOMapper.insert(clubDO);

            // 插入主主持人到 t_club_host 表
            ClubHostDO clubHostDO = ClubHostDO.builder()
                    .clubId(clubDO.getId())
                    .userId(creatorId)
                    .hostLevel(1) // 主主持人等级设为 1
                    .createTime(LocalDateTime.now())
                    .updateTime(LocalDateTime.now())
                    .build();
            clubHostMapper.insert(clubHostDO);

        } catch (Exception e) {
            log.error("==> 创建俱乐部失败", e);
            // 如果创建失败，抛出异常
            throw new BizException(ResponseCodeEnum.CLUB_CREATION_FAILED);
        }

        // 返回创建成功的俱乐部 ID
        return Response.success(clubDO.getId());
    }


    /**
     * 更新俱乐部信息
     *
     * @param clubId 俱乐部 ID
     * @param name 新的俱乐部名称
     * @param description 新的俱乐部描述
     * @return 更新后的俱乐部信息
     */
    @Override
    @Transactional
    public Response<UpdatedClubResponseDTO> updateClub(Long clubId, String name, String description) {
        // 校验俱乐部名称是否为空
        if (StringUtils.isBlank(name)) {
            throw new BizException(ResponseCodeEnum.CLUB_NAME_EMPTY); // 若名称为空则抛出异常
        }

        // 获取当前用户的 ID
        Long userId = LoginUserContextHolder.getUserId();
        if (userId == null) {
            throw new BizException(ResponseCodeEnum.ID_NOT_VALID);
        }

        // 查找俱乐部信息
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null|| clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 校验用户权限（第一主持人或第二主持人）
        if (!userId.equals(clubDO.getCreatorId()) && !isSecondHost(userId, clubId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 用户非主持人，抛出权限不足异常
        }

        // 更新俱乐部信息
        clubDO.setName(name);
        clubDO.setDescription(description);
        clubDO.setUpdateTime(LocalDateTime.now());

        // 保存更新
        clubDOMapper.updateByPrimaryKeySelective(clubDO);

        // 返回更新后的简化信息
        UpdatedClubResponseDTO responseDTO = new UpdatedClubResponseDTO(
                clubDO.getId(),
                clubDO.getName(),
                clubDO.getDescription(),
                clubDO.getUpdateTime()
        );

        return Response.success(responseDTO);
    }


    /**
     * 校验是否为第二主持人
     *
     * @param userId 用户 ID
     * @param clubId 俱乐部 ID
     * @return 是否为第二主持人
     */
    private boolean isSecondHost(Long userId, Long clubId) {
        // 查询俱乐部主持人表以判断用户是否为第二主持人
        return clubHostMapper.isSecondHost(userId, clubId);
    }

    /**
     * 删除俱乐部（逻辑删除）
     *
     * @param clubId 俱乐部 ID
     * @return 删除操作的响应
     */
    @Override
    @Transactional
    public Response<?> deleteClub(Long clubId) {
        // 获取当前用户的 ID
        Long userId = LoginUserContextHolder.getUserId();
        if (userId == null) {
            throw new BizException(ResponseCodeEnum.ID_NOT_VALID);
        }

        // 查找俱乐部信息
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 校验用户是否为第一主持人
        if (!userId.equals(clubDO.getCreatorId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 只有第一主持人可以删除
        }

        // 设置逻辑删除标志
        clubDO.setIsDeleted(true);

        // 更新数据库
        clubDOMapper.updateByPrimaryKeySelective(clubDO);

        // 返回成功响应
        return Response.success("俱乐部已成功删除");
    }

    /**
     * 获取俱乐部详情
     *
     * @param clubId 俱乐部 ID
     * @return 俱乐部详情
     */
    @Override
    public Response<ClubDetailsDTO> getClubDetails(Long clubId) {
        // 查找俱乐部信息
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在或已删除，抛出异常
        }

        // 构建并返回俱乐部详情
        ClubDetailsDTO detailsDTO = new ClubDetailsDTO(
                clubDO.getId(),
                clubDO.getName(),
                clubDO.getDescription(),
                clubDO.getCreatorId(),
                clubDO.getCreateTime(),
                clubDO.getUpdateTime()
        );

        return Response.success(detailsDTO);
    }

    @Override
    @Transactional
    public Response<ClubListDTO> getClubList(int page, int size) {
        // 设置分页查询的起始点和数量
        int offset = (page - 1) * size;

        // 查询总数
        long total = clubDOMapper.countAll();

        // 分页查询俱乐部信息
        List<ClubDO> clubList = clubDOMapper.selectAllPaginated(offset, size);

        // 转换为 ClubListDTO.ClubItemDTO 格式
        List<ClubListDTO.ClubItemDTO> clubs = clubList.stream().map(club -> {
            ClubListDTO.ClubItemDTO item = new ClubListDTO.ClubItemDTO();
            item.setClubId(club.getId());
            item.setName(club.getName());
            item.setDescription(club.getDescription());
            item.setCreateTime(club.getCreateTime());
            return item;
        }).collect(Collectors.toList());

        // 创建并返回 ClubListDTO
        ClubListDTO responseDTO = new ClubListDTO();
        responseDTO.setClubs(clubs);
        responseDTO.setCurrentPage(page);
        responseDTO.setPageSize(size);
        responseDTO.setTotal(total);

        return Response.success(responseDTO);
    }

    /**
     * 添加第二主持人
     *
     * @param clubId 俱乐部 ID
     * @param userId 用户 ID（将成为第二主持人）
     * @return 操作结果
     */
    @Override
    @Transactional
    public Response<?> addSecondHost(Long clubId, Long userId) {
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 校验用户是否为第一主持人
        if (!currentUserId.equals(clubDO.getCreatorId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 检查用户是否存在
        UserDO user = userDOMapper.selectByPrimaryKey(userId);
        if (user == null) {
            throw new BizException(ResponseCodeEnum.USER_NOT_FOUND); // 用户不存在
        }

        // 检查用户是否已经是该俱乐部的主持人
        ClubHostDO existingHost = clubHostMapper.selectByClubIdAndUserId(clubId, userId);
        if (existingHost != null) {
            throw new BizException(ResponseCodeEnum.ALREADY_A_HOST); // 用户已是主持人
        }

        // 检查用户是否已经是俱乐部成员，若不是则自动添加
        ClubMemberDO existingMember = clubMemberDOMapper.selectByClubIdAndUserId(clubId, userId);

        // 调用分布式 ID 生成服务，生成成员 ID
        String memberId = distributedIdGeneratorRpcService.getSnowflakeId();

        if (existingMember == null) {
            ClubMemberDO newMember = ClubMemberDO.builder()
                    .id(Long.valueOf(memberId))
                    .clubId(clubId)
                    .userId(userId)
                    .joinTime(LocalDateTime.now()) // 设置加入时间
                    .build();
            clubMemberDOMapper.insert(newMember);
        }

        // 调用分布式 ID 生成服务，生成主持人 ID
        String hostId = distributedIdGeneratorRpcService.getSnowflakeId();

        // 添加第二主持人
        ClubHostDO newHost = ClubHostDO.builder()
                .id(Long.valueOf(hostId))
                .clubId(clubId)
                .userId(userId)
                .hostLevel(2) // 设置为第二主持人
                .createTime(LocalDateTime.now()) // 设置创建时间
                .updateTime(LocalDateTime.now()) // 设置更新时间
                .build();
        clubHostMapper.insert(newHost);

        return Response.success("第二主持人添加成功");
    }

    /**
     * 移除主持人身份
     *
     * @param clubId 俱乐部ID
     * @param userId 要移除或转让的用户ID
     * @return 操作结果
     */
    @Override
    @Transactional
    public Response<?> removeHost(Long clubId, Long userId) {
        Long currentUserId = LoginUserContextHolder.getUserId();

        ClubHostDO primaryHost = clubHostMapper.selectFirstHostByClubId(clubId);

        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);

        // 校验用户是否为第一主持人
        if (!currentUserId.equals(clubDO.getCreatorId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 查找俱乐部信息
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在或已删除，抛出异常
        }
        // 检查目标用户是否存在于俱乐部中
        ClubHostDO targetHost = clubHostMapper.selectByClubIdAndUserId(clubId, userId);
        if (targetHost == null) {
            throw new BizException(ResponseCodeEnum.HOST_NOT_FOUND); // 指定的主持人不存在
        }
        // 移除第二主持人
        clubHostMapper.deleteByPrimaryKey(targetHost.getId());

        return Response.success("主持人移除成功");
    }

    /**
     * 获取指定俱乐部的主持人列表
     *
     * @param clubId 俱乐部 ID
     * @return 主持人列表
     */
    @Override
    @Transactional(readOnly = true)
    public Response<List<ClubHostDTO>> getHosts(Long clubId) {
        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 查询俱乐部的主持人列表
        List<ClubHostDO> hostList = clubHostMapper.selectByClubId(clubId);
        if (hostList.isEmpty()) {
            return Response.success(new ArrayList<>()); // 若无主持人，返回空列表
        }

        // 将 ClubHostDO 转换为 ClubHostDTO
        List<ClubHostDTO> hostDTOList = hostList.stream().map(host -> new ClubHostDTO(
                host.getUserId(),
                host.getHostLevel(),
                host.getCreateTime(),
                host.getUpdateTime()
        )).collect(Collectors.toList());

        return Response.success(hostDTOList);
    }

    /**
     * 用户申请加入俱乐部
     *
     * @param requestVO 包含 clubId 的 VO 对象
     * @return 操作结果
     */
    @Override
    @Transactional
    public Response<?> requestToJoinClub(JoinClubReqVO requestVO) {
        Long userId = LoginUserContextHolder.getUserId();
        Long clubId = requestVO.getClubId();

        // 检查是否已有申请记录
        ClubJoinReqDO existingRequest = clubJoinReqMapper.selectByClubIdAndUserId(clubId, userId);

        if (existingRequest != null) {
            // 处理不同状态
            if (existingRequest.getStatus() == 0) {
                throw new BizException(ResponseCodeEnum.DUPLICATE_JOIN_REQUEST); // 待审核状态
            } else if (existingRequest.getStatus() == 1) {
                return Response.success("您已是该俱乐部成员"); // 已通过状态
            } else if (existingRequest.getStatus() == 2 || existingRequest.getStatus() == 3) {
                // 状态为已拒绝或已退出俱乐部，更新状态为待审核重新提交
                existingRequest.setStatus(0); // 重新设置为待审核状态
                existingRequest.setUpdateTime(LocalDateTime.now());
                clubJoinReqMapper.updateByPrimaryKeySelective(existingRequest);
                return Response.success("重新提交加入申请成功");
            }
        } else {
            // 若无历史申请记录，插入新的申请记录
            String submitId = distributedIdGeneratorRpcService.getSnowflakeId();
            ClubJoinReqDO joinRequest = ClubJoinReqDO.builder()
                    .id(Long.valueOf(submitId))
                    .clubId(clubId)
                    .userId(userId)
                    .status(0) // 待审核状态
                    .createTime(LocalDateTime.now())
                    .updateTime(LocalDateTime.now())
                    .build();
            clubJoinReqMapper.insert(joinRequest);
        }

        return Response.success("申请已提交");
    }


    /**
     * 审核用户申请加入俱乐部
     *
     * @param reviewJoinReqVO 审核请求对象，包含俱乐部ID、用户ID和审核状态
     * @return 操作结果
     */
    @Override
    @Transactional
    public Response<?> reviewJoinRequest(ReviewJoinReqVO reviewJoinReqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long clubId = reviewJoinReqVO.getClubId();
        Long userId = reviewJoinReqVO.getUserId();
        int status = reviewJoinReqVO.getStatus();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 校验当前用户是否为第一或第二主持人
        if (!currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 用户非主持人，抛出权限不足异常
        }

        // 查询加入申请
        ClubJoinReqDO joinRequest = clubJoinReqMapper.selectByClubIdAndUserId(clubId, userId);
        if (joinRequest == null) {
            throw new BizException(ResponseCodeEnum.REQUEST_NOT_FOUND); // 没有找到申请
        }

        // 检查申请是否已经审核过
        if (joinRequest.getStatus() == 1 || joinRequest.getStatus() == 2) {
            return Response.success("该申请已审核"); // 申请已审核过，返回提示信息
        }

        // 检查审核状态是否有效
        if (status != 1 && status != 2) {
            throw new BizException(ResponseCodeEnum.INVALID_OPERATION); // 仅允许同意(1)或拒绝(2)
        }

        // 更新申请状态
        joinRequest.setStatus(status);
        joinRequest.setUpdateTime(LocalDateTime.now());
        clubJoinReqMapper.updateByPrimaryKeySelective(joinRequest);

        // 如果审核状态为同意，将用户加入俱乐部成员表
        if (status == 1) {
            ClubMemberDO existingMember = clubMemberDOMapper.selectByClubIdAndUserId(clubId, userId);

            if (existingMember == null) { // 若用户尚未是俱乐部成员
                // 调用分布式 ID 生成服务，生成成员 ID
                String memberId = distributedIdGeneratorRpcService.getSnowflakeId();

                ClubMemberDO newMember = ClubMemberDO.builder()
                        .id(Long.valueOf(memberId))
                        .clubId(clubId)
                        .userId(userId)
                        .joinTime(LocalDateTime.now())
                        .build();
                clubMemberDOMapper.insert(newMember);
            }
            return Response.success("已同意");
        } else if (status == 2) {
            return Response.success("已拒绝");
        }

        // 返回响应
        return Response.success("未知的操作");
    }


    /**
     * 用户退出俱乐部
     *
     * @param clubId 俱乐部ID
     * @return 退出操作结果
     */
    @Override
    @Transactional
    public Response<?> leaveClub(Long clubId) {
        // 获取当前用户ID
        Long userId = LoginUserContextHolder.getUserId();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 检查用户是否为俱乐部成员
        ClubMemberDO member = clubMemberDOMapper.selectByClubIdAndUserId(clubId, userId);
        if (member == null) {
            throw new BizException(ResponseCodeEnum.MEMBER_NOT_FOUND); // 用户不是俱乐部成员
        }

        // 检查用户是否为主持人
        ClubHostDO host = clubHostMapper.selectByClubIdAndUserId(clubId, userId);
        if (host != null) {
            throw new BizException(ResponseCodeEnum.HOST_CANNOT_LEAVE); // 主持人无法直接退出
        }

        // 删除用户的俱乐部成员记录
        clubMemberDOMapper.deleteByPrimaryKey(member.getId());

        // 重置用户的加入请求状态
        ClubJoinReqDO joinRequest = clubJoinReqMapper.selectByClubIdAndUserId(clubId, userId);
        if (joinRequest != null && joinRequest.getStatus() == 1) { // 如果已通过，则重置状态
            joinRequest.setStatus(3); //已退出
            joinRequest.setUpdateTime(LocalDateTime.now());
            clubJoinReqMapper.updateByPrimaryKeySelective(joinRequest);
        }

        return Response.success("成功退出俱乐部");
    }

    /**
     * 获取用户申请加入俱乐部的记录列表，支持分页
     *
     * @param page 页码
     * @param size 每页记录数
     * @return 分页的加入申请记录列表
     */
    @Override
    @Transactional(readOnly = true)
    public Response<List<JoinRequestDTO>> getJoinRequestList(int page, int size) {
        Long userId = LoginUserContextHolder.getUserId();

        int offset = (page - 1) * size;

        // 查询用户的加入俱乐部申请记录
        List<ClubJoinReqDO> joinRequestList = clubJoinReqMapper.selectByUserIdWithPagination(userId, offset, size);

        // 将结果转换为 DTO 列表
        List<JoinRequestDTO> joinRequestDTOList = joinRequestList.stream().map(joinRequest ->
                new JoinRequestDTO(
                        joinRequest.getClubId(),
                        joinRequest.getUserId(),
                        joinRequest.getStatus(),
                        joinRequest.getCreateTime(),
                        joinRequest.getUpdateTime()
                )
        ).collect(Collectors.toList());

        return Response.success(joinRequestDTOList);
    }

    @Override
    @Transactional(readOnly = true)
    public Response<List<JoinRequestDTO>> getClubJoinRequests(Long clubId, int page, int size) {
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND); // 若俱乐部不存在，抛出异常
        }

        // 验证当前用户是否为第一或第二主持人
        if (!currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 用户非主持人，抛出权限不足异常
        }

        // 设置分页参数
        int offset = (page - 1) * size;

        // 查询俱乐部的加入申请记录
        List<ClubJoinReqDO> joinRequests = clubJoinReqMapper.selectByClubIdWithPagination(clubId, offset, size);

        // 将结果转换为 DTO 列表
        List<JoinRequestDTO> joinRequestDTOList = joinRequests.stream().map(joinRequest ->
                new JoinRequestDTO(
                        joinRequest.getClubId(),
                        joinRequest.getUserId(),
                        joinRequest.getStatus(),
                        joinRequest.getCreateTime(),
                        joinRequest.getUpdateTime()
                )
        ).collect(Collectors.toList());

        return Response.success(joinRequestDTOList);
    }

    @Override
    @Transactional(readOnly = true)
    public Response<List<ClubMemberDTO>> getClubMembers(Long clubId, int page, int size) {
        // 验证俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND);
        }

        int offset = (page - 1) * size;

        // 查询俱乐部成员列表，获取用户的昵称、头像等信息
        List<ClubMemberDTO> members = clubMemberDOMapper.selectByClubIdWithPagination(clubId, offset, size);

        return Response.success(members);
    }

    @Override
    @Transactional
    public Response<Long> publishActivity(PublishActivityReqVO publishActivityReqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long clubId = publishActivityReqVO.getClubId();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || clubDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND);
        }

        // 验证用户是否为俱乐部的主持人
        if (!currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 创建活动ID
        String activityId = distributedIdGeneratorRpcService.getSnowflakeId();

        // 插入活动信息
        ActivityDO activityDO = ActivityDO.builder()
                .id(Long.valueOf(activityId))
                .clubId(clubId)
                .hostId(currentUserId)
                .title(publishActivityReqVO.getTitle())
                .content(publishActivityReqVO.getContent())
                .startTime(publishActivityReqVO.getStartTime())
                .endTime(publishActivityReqVO.getEndTime())
                .createTime(LocalDateTime.now())
                .updateTime(LocalDateTime.now())
                .isDeleted(false)
                .build();

        activityDOMapper.insert(activityDO);

        return Response.success(activityDO.getId());
    }

    //更新活动
    @Override
    @Transactional
    public Response<?> updateActivity(UpdateActivityReqVO updateActivityReqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long activityId = updateActivityReqVO.getActivityId();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null|| Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND); // 若活动不存在，抛出异常
        }

        // 验证当前用户是否为活动所属俱乐部的主持人
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(activityDO.getClubId());
        if (clubDO == null || !currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubDO.getId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 用户非主持人，抛出权限不足异常
        }

        // 判断 startTime 和 endTime 是否一起更新
        if ((updateActivityReqVO.getStartTime() != null && updateActivityReqVO.getEndTime() == null) ||
                (updateActivityReqVO.getStartTime() == null && updateActivityReqVO.getEndTime() != null)) {
            throw new BizException(ResponseCodeEnum.INVALID_OPERATION);
        }

        // 更新非空字段，避免覆盖为空的值
        ActivityDO updatedActivityDO = new ActivityDO();
        updatedActivityDO.setId(activityId); // 设置主键
        if (updateActivityReqVO.getTitle() != null) {
            updatedActivityDO.setTitle(updateActivityReqVO.getTitle());
        }
        if (updateActivityReqVO.getContent() != null) {
            updatedActivityDO.setContent(updateActivityReqVO.getContent());
        }
        if (updateActivityReqVO.getStartTime() != null && updateActivityReqVO.getEndTime() != null) {
            updatedActivityDO.setStartTime(updateActivityReqVO.getStartTime());
            updatedActivityDO.setEndTime(updateActivityReqVO.getEndTime());
        }

        // 更新活动的更新时间
        updatedActivityDO.setUpdateTime(LocalDateTime.now());
        activityDOMapper.updateByPrimaryKeySelective(updatedActivityDO);

        return Response.success("活动已更新");
    }

    @Override
    @Transactional
    public Response<?> deleteActivity(DeleteActivityReqVO deleteActivityReqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long clubId = deleteActivityReqVO.getClubId();
        Long activityId = deleteActivityReqVO.getActivityId();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || activityDO.getIsDeleted()) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 验证活动是否属于指定俱乐部
        if (!activityDO.getClubId().equals(clubId)) {
            throw new BizException(ResponseCodeEnum.INVALID_OPERATION); // 活动不属于指定的俱乐部
        }

        // 验证当前用户是否为活动所属俱乐部的主持人
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(activityDO.getClubId());
        if (clubDO == null || !currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubDO.getId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED); // 用户非主持人，抛出权限不足异常
        }

        // 逻辑删除活动
        activityDO.setIsDeleted(true);
        activityDO.setUpdateTime(LocalDateTime.now());
        activityDOMapper.updateByPrimaryKeySelective(activityDO);

        return Response.success("活动已删除");
    }

    @Override
    @Transactional(readOnly = true)
    public Response<ActivityDetailsDTO> getActivityDetails(Long activityId) {
        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 获取活动的俱乐部信息
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(activityDO.getClubId());
        if (clubDO == null || Boolean.TRUE.equals(clubDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND);
        }

        // 转换活动信息到 DTO
        ActivityDetailsDTO activityDetailsDTO = ActivityDetailsDTO.builder()
                .activityId(activityDO.getId())
                .clubId(activityDO.getClubId())
                .title(activityDO.getTitle())
                .content(activityDO.getContent())
                .startTime(activityDO.getStartTime())
                .endTime(activityDO.getEndTime())
                .createTime(activityDO.getCreateTime())
                .updateTime(activityDO.getUpdateTime())
                .build();

        return Response.success(activityDetailsDTO);
    }

    @Override
    @Transactional(readOnly = true)
    public Response<List<ActivityDetailsDTO>> getClubActivityList(Long clubId, int page, int size) {
        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || Boolean.TRUE.equals(clubDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND);
        }

        // 设置分页参数
        int offset = (page - 1) * size;

        // 查询活动列表
        List<ActivityDO> activityDOList = activityDOMapper.selectActiveByClubIdWithPagination(clubId, offset, size);

        // 转换活动信息到 DTO 列表
        List<ActivityDetailsDTO> activityDetailsDTOList = activityDOList.stream()
                .map(activityDO -> ActivityDetailsDTO.builder()
                        .activityId(activityDO.getId())
                        .clubId(activityDO.getClubId())
                        .title(activityDO.getTitle())
                        .content(activityDO.getContent())
                        .startTime(activityDO.getStartTime())
                        .endTime(activityDO.getEndTime())
                        .createTime(activityDO.getCreateTime())
                        .updateTime(activityDO.getUpdateTime())
                        .build())
                .collect(Collectors.toList());

        return Response.success(activityDetailsDTOList);
    }

    @Override
    @Transactional
    public Response<?> signupForActivity(Long activityId) {
        Long userId = LoginUserContextHolder.getUserId();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 检查用户是否为活动所属俱乐部的成员
        ClubMemberDO member = clubMemberDOMapper.selectByClubIdAndUserId(activityDO.getClubId(), userId);
        if (member == null) {
            throw new BizException(ResponseCodeEnum.MEMBER_NOT_FOUND); // 用户不是俱乐部成员，抛出异常
        }

        // 检查是否已有报名记录
        ActivitySignupDO existingSignup = activitySignupDOMapper.selectByActivityIdAndUserId(activityId, userId);
        if (existingSignup != null) {
            if (existingSignup.getStatus() == 0) {
                throw new BizException(ResponseCodeEnum.ALREADY_APPLIED); // 已提交，待审核状态
            } else if (existingSignup.getStatus() == 1) {
                return Response.success("您已报名成功"); // 报名通过状态
            } else if (existingSignup.getStatus() == 2) {
                existingSignup.setStatus(0); // 重置为待审核状态
                existingSignup.setUpdateTime(LocalDateTime.now());
                activitySignupDOMapper.updateByPrimaryKeySelective(existingSignup);
                return Response.success("重新提交报名申请成功");
            } else if (existingSignup.getStatus() == 3) {
                existingSignup.setStatus(0); // 重置为待审核状态
                existingSignup.setUpdateTime(LocalDateTime.now());
                activitySignupDOMapper.updateByPrimaryKeySelective(existingSignup);
                return Response.success("重新提交报名申请成功");
            }
        } else {
            // 调用分布式 ID 生成服务，生成报名记录 ID
            String signupId = distributedIdGeneratorRpcService.getSnowflakeId();

            // 创建新的报名记录
            ActivitySignupDO signup = ActivitySignupDO.builder()
                    .id(Long.valueOf(signupId))
                    .activityId(activityId)
                    .userId(userId)
                    .signupTime(LocalDateTime.now())
                    .status(0) // 初始状态为待审核
                    .build();
            activitySignupDOMapper.insert(signup);
        }

        return Response.success("报名申请已提交");
    }

    @Override
    @Transactional
    public Response<?> reviewSignupRequest(ReviewSignupReqVO reviewSignupReqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long activityId = reviewSignupReqVO.getActivityId();
        Long userId = reviewSignupReqVO.getUserId();
        Integer status = reviewSignupReqVO.getStatus();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 验证当前用户是否为活动的主持人
        if (!currentUserId.equals(activityDO.getHostId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 查询报名申请记录
        ActivitySignupDO signupRequest = activitySignupDOMapper.selectByActivityIdAndUserId(activityId, userId);
        if (signupRequest == null) {
            throw new BizException(ResponseCodeEnum.REQUEST_NOT_FOUND); // 没有找到申请
        }

        // 检查申请是否已经审核过
        if (signupRequest.getStatus() == 1 || signupRequest.getStatus() == 2) {
            return Response.success("该申请已审核"); // 申请已审核过
        }

        // 检查审核状态是否合法（1表示同意，2表示拒绝）
        if (status != 1 && status != 2) {
            throw new BizException(ResponseCodeEnum.INVALID_OPERATION); // 状态无效
        }

        // 更新报名申请状态
        signupRequest.setStatus(status);
        signupRequest.setUpdateTime(LocalDateTime.now());
        activitySignupDOMapper.updateByPrimaryKeySelective(signupRequest);

        // 如果审核通过，将用户记录到活动成员表
        // 调用分布式 ID 生成服务，活动成员记录 ID
        String actmemId = distributedIdGeneratorRpcService.getSnowflakeId();
        if (status == 1) {
            ActivityMemberDO activityMember = ActivityMemberDO.builder()
                    .id(Long.valueOf(actmemId))
                    .activityId(activityId)
                    .userId(userId)
                    .joinTime(LocalDateTime.now())
                    .build();
            activityMemberDOMapper.insert(activityMember);
            return Response.success("已同意报名并加入活动成员");
        } else {
            return Response.success("已拒绝报名");
        }
    }

    @Override
    @Transactional
    public Response<?> cancelSignup(CancelSignupReqVO cancelSignupReqVO) {
        Long userId = cancelSignupReqVO.getUserId();
        Long activityId = cancelSignupReqVO.getActivityId();

        // 检查报名申请是否存在且状态为已通过（status = 1）
        ActivitySignupDO signupRecord = activitySignupDOMapper.selectByActivityIdAndUserId(activityId, userId);
        if (signupRecord == null || signupRecord.getStatus() != 1) {
            throw new BizException(ResponseCodeEnum.REQUEST_NOT_FOUND); // 报名记录不存在或未通过审核
        }

        // 更新报名申请的状态为3（取消报名）
        signupRecord.setStatus(3);
        signupRecord.setUpdateTime(LocalDateTime.now());
        activitySignupDOMapper.updateByPrimaryKeySelective(signupRecord);

        // 从活动成员表中删除该用户
        activityMemberDOMapper.deleteByActivityIdAndUserId(activityId, userId);

        return Response.success("报名已取消");
    }

    @Override
    @Transactional
    public Response<?> removeMemberFromActivity(RemoveMemberReqVO removeMemberReqVO) {
        Long activityId = removeMemberReqVO.getActivityId();
        Long userId = removeMemberReqVO.getUserId();
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 验证当前用户是否为活动的主持人
        if (!currentUserId.equals(activityDO.getHostId())) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 删除成员的活动报名记录
        int deletedRows = activityMemberDOMapper.deleteByActivityIdAndUserId(activityId, userId);
        if (deletedRows == 0) {
            throw new BizException(ResponseCodeEnum.MEMBER_NOT_FOUND_IN_ACTIVITY); // 该成员不在活动中
        }

        // 更新报名状态为取消（3）
        ActivitySignupDO signupRecord = activitySignupDOMapper.selectByActivityIdAndUserId(activityId, userId);
        if (signupRecord != null) {
            signupRecord.setStatus(3);
            signupRecord.setUpdateTime(LocalDateTime.now());
            activitySignupDOMapper.updateByPrimaryKeySelective(signupRecord);
        }

        return Response.success("成员已从活动中移除");
    }

    @Override
    @Transactional
    public Response<?> removeMemberFromClub(RemoveMemberReqVO removeMemberReqVO) {
        Long clubId = removeMemberReqVO.getClubId();
        Long userId = removeMemberReqVO.getUserId();
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 检查俱乐部是否存在且未被删除
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(clubId);
        if (clubDO == null || Boolean.TRUE.equals(clubDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.CLUB_NOT_FOUND);
        }

        // 验证当前用户是否为俱乐部的主持人
        if (!currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 删除俱乐部成员记录
        int deletedRows = clubMemberDOMapper.deleteByClubIdAndUserId(clubId, userId);
        if (deletedRows == 0) {
            throw new BizException(ResponseCodeEnum.MEMBER_NOT_FOUND_IN_CLUB); // 该成员不在俱乐部中
        }

        // 更新加入请求表中的状态为已退出（3）
        ClubJoinReqDO joinRequest = clubJoinReqMapper.selectByClubIdAndUserId(clubId, userId);
        if (joinRequest != null) {
            joinRequest.setStatus(3); // 设置为已退出状态
            joinRequest.setUpdateTime(LocalDateTime.now());
            clubJoinReqMapper.updateByPrimaryKeySelective(joinRequest);
        }

        // 删除该成员在俱乐部内所有活动的报名记录
        List<ActivitySignupDO> signupRecords = activitySignupDOMapper.selectByClubIdAndUserId(clubId, userId);
        for (ActivitySignupDO signupRecord : signupRecords) {
            signupRecord.setStatus(3); // 设置状态为取消
            signupRecord.setUpdateTime(LocalDateTime.now());
            activitySignupDOMapper.updateByPrimaryKeySelective(signupRecord);

            // 从活动成员表中移除该成员
            activityMemberDOMapper.deleteByActivityIdAndUserId(signupRecord.getActivityId(), userId);
        }

        return Response.success("成员已从俱乐部及相关活动中移除");
    }

    @Override
    @Transactional(readOnly = true)
    public Response<List<ActivityMemberDTO>> getActivitySignupMembers(ActivitySignupMembersReqVO reqVO) {
        Long activityId = reqVO.getActivityId();
        int offset = (reqVO.getPage() - 1) * reqVO.getSize();
        int limit = reqVO.getSize();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 查询报名成员列表
        List<ActivityMemberDTO> members = activitySignupDOMapper.selectMembersByActivityId(activityId, offset, limit);

        return Response.success(members);
    }

    @Override
    @Transactional(readOnly = true)
    public Response<List<ActivitySignupInfoDTO>> getUserActivitySignups(UserActivitySignupListReqVO reqVO) {
        Long userId = LoginUserContextHolder.getUserId();
        int offset = (reqVO.getPage() - 1) * reqVO.getSize();
        int limit = reqVO.getSize();

        // 查询用户报名申请列表
        List<ActivitySignupInfoDTO> signups = activitySignupDOMapper.selectSignupsByUserId(userId, offset, limit);

        return Response.success(signups);
    }
    @Override
    @Transactional(readOnly = true)
    public Response<List<ActivitySignupDetailDTO>> getActivitySignupsForHost(HostActivitySignupListReqVO reqVO) {
        Long currentUserId = LoginUserContextHolder.getUserId();
        Long activityId = reqVO.getActivityId();
        int offset = (reqVO.getPage() - 1) * reqVO.getSize();
        int limit = reqVO.getSize();

        // 检查活动是否存在且未被删除
        ActivityDO activityDO = activityDOMapper.selectByPrimaryKey(activityId);
        if (activityDO == null || Boolean.TRUE.equals(activityDO.getIsDeleted())) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 验证当前用户是否为活动的主持人
        ClubDO clubDO = clubDOMapper.selectByPrimaryKey(activityDO.getClubId());
        if (clubDO == null || (!currentUserId.equals(clubDO.getCreatorId()) && !isSecondHost(currentUserId, clubDO.getId()))) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 查询活动报名申请列表
        List<ActivitySignupDetailDTO> signups = activitySignupDOMapper.selectSignupsByActivityId(activityId, offset, limit);

        return Response.success(signups);
    }



}
