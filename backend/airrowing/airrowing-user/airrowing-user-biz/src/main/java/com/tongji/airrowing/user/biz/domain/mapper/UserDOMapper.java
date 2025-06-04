package com.tongji.airrowing.user.biz.domain.mapper;

import com.tongji.airrowing.user.biz.domain.dataobject.UserDO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface UserDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(UserDO record);

    int insertSelective(UserDO record);

    UserDO selectByPrimaryKey(Long id);


    int updateByPrimaryKeySelective(UserDO record);

    int updateByPrimaryKey(UserDO record);

    /**
     * 批量查询用户信息
     *
     * @param ids
     * @return
     */
    List<UserDO> selectByIds(@Param("ids") List<Long> ids);

    /**
     * 根据邮箱查询记录
     * @param email
     * @return
     */
    UserDO selectByEmail(String email);

    /**
     * 根据昵称查询记录
     * @param email
     * @return
     */
    UserDO selectByNickname(String email);
}