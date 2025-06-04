package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.ActivitySignupDO;
import com.tongji.airrowing.club.biz.model.dto.ActivityMemberDTO;
import com.tongji.airrowing.club.biz.model.dto.ActivitySignupDetailDTO;
import com.tongji.airrowing.club.biz.model.dto.ActivitySignupInfoDTO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface ActivitySignupDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(ActivitySignupDO record);

    int insertSelective(ActivitySignupDO record);

    ActivitySignupDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(ActivitySignupDO record);

    int updateByPrimaryKey(ActivitySignupDO record);

    ActivitySignupDO selectByActivityIdAndUserId(@Param("activityId") Long activityId, @Param("userId") Long userId);
    List<ActivitySignupDO> selectByClubIdAndUserId(@Param("clubId") Long clubId, @Param("userId") Long userId);

    List<ActivityMemberDTO> selectMembersByActivityId(@Param("activityId") Long activityId,
                                                      @Param("offset") int offset,
                                                      @Param("limit") int limit);

    List<ActivitySignupInfoDTO> selectSignupsByUserId(@Param("userId") Long userId, @Param("offset") int offset, @Param("limit") int limit);

    List<ActivitySignupDetailDTO> selectSignupsByActivityId(@Param("activityId") Long activityId, @Param("offset") int offset, @Param("limit") int limit);

}
