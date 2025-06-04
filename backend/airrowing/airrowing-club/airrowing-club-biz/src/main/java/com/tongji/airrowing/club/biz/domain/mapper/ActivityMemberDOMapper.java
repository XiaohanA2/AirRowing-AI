package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ActivityMemberDO;
import org.apache.ibatis.annotations.Param;

public interface ActivityMemberDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ActivityMemberDO record);

    int insertSelective(ActivityMemberDO record);

    ActivityMemberDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ActivityMemberDO record);

    int updateByPrimaryKey(ActivityMemberDO record);

    int deleteByActivityIdAndUserId(@Param("activityId") Long activityId, @Param("userId") Long userId);
}