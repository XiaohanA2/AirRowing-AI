package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ClubHostDO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ClubHostDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ClubHostDO record);

    int insertSelective(ClubHostDO record);

    ClubHostDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ClubHostDO record);

    int updateByPrimaryKey(ClubHostDO record);

    /**
     * 检查用户是否为第二主持人
     *
     * @param userId 用户 ID
     * @param clubId 俱乐部 ID
     * @return 是否为第二主持人
     */
    boolean isSecondHost(@Param("userId") Long userId, @Param("clubId") Long clubId);

    /**
     * 查询指定俱乐部的第一主持人
     *
     * @param clubId 俱乐部 ID
     * @return 该俱乐部的第一主持人记录，若不存在则返回 null
     */
    ClubHostDO selectFirstHostByClubId(@Param("clubId") Long clubId);

    /**
     * 查询指定俱乐部和用户的主持人记录
     *
     * @param clubId 俱乐部 ID
     * @param userId 用户 ID
     * @return 若用户是该俱乐部的主持人，返回主持人记录；否则返回 null
     */
    ClubHostDO selectByClubIdAndUserId(@Param("clubId") Long clubId, @Param("userId") Long userId);

    List<ClubHostDO> selectByClubId(Long clubId);

}