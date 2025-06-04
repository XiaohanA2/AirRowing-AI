package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ClubJoinReqDO;
import com.tongji.airrowing.club.biz.model.dto.JoinRequestDTO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ClubJoinReqDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ClubJoinReqDO record);

    int insertSelective(ClubJoinReqDO record);

    ClubJoinReqDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ClubJoinReqDO record);

    int updateByPrimaryKey(ClubJoinReqDO record);

    // 根据 clubId 和 userId 查询加入请求记录
    ClubJoinReqDO selectByClubIdAndUserId(@Param("clubId") Long clubId, @Param("userId") Long userId);

    // 根据 userId 查询加入申请记录并支持分页
    List<ClubJoinReqDO> selectByUserIdWithPagination(@Param("userId") Long userId, @Param("offset") int offset, @Param("size") int size);

    List<ClubJoinReqDO> selectByClubIdWithPagination(@Param("clubId") Long clubId, @Param("offset") int offset, @Param("size") int size);


}