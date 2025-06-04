package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ClubMemberDO;
import com.tongji.airrowing.club.biz.model.dto.ClubMemberDTO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ClubMemberDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ClubMemberDO record);

    int insertSelective(ClubMemberDO record);

    ClubMemberDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ClubMemberDO record);

    int updateByPrimaryKey(ClubMemberDO record);

    ClubMemberDO selectByClubIdAndUserId(@Param("clubId") Long clubId, @Param("userId") Long userId);

    List<ClubMemberDTO> selectByClubIdWithPagination(@Param("clubId") Long clubId, @Param("offset") int offset, @Param("size") int size);

    int deleteByClubIdAndUserId(@Param("clubId") Long clubId, @Param("userId") Long userId);
}