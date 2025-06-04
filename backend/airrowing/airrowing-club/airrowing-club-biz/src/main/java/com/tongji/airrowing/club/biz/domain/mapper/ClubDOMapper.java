package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ClubDO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ClubDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ClubDO record);

    int insertSelective(ClubDO record);

    ClubDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ClubDO record);

    int updateByPrimaryKey(ClubDO record);

    long countAll();

    List<ClubDO> selectAllPaginated(@Param("offset") int offset, @Param("limit") int limit);
}