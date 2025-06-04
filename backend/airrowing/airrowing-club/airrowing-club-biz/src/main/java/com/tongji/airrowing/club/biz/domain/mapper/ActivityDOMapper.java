package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ActivityDO;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

public interface ActivityDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ActivityDO record);

    int insertSelective(ActivityDO record);

    ActivityDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ActivityDO record);

    int updateByPrimaryKeyWithBLOBs(ActivityDO record);

    int updateByPrimaryKey(ActivityDO record);

    int logicalDeleteById(@Param("id") Long id, @Param("updateTime") LocalDateTime updateTime);

    ActivityDO selectActiveById(@Param("id") Long id); // 查询未被逻辑删除的活动

    List<ActivityDO> selectActiveByClubIdWithPagination(@Param("clubId") Long clubId, @Param("offset") int offset, @Param("size") int size);

}