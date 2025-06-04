package com.tongji.airrowing.club.biz.domain.mapper;

import com.tongji.airrowing.club.biz.domain.dataobject.TrainingDataDO;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

public interface TrainingDataDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(TrainingDataDO record);

    int insertSelective(TrainingDataDO record);

    TrainingDataDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(TrainingDataDO record);

    int updateByPrimaryKey(TrainingDataDO record);

    /**
     * 查询用户的训练数据，支持分页和筛选条件
     *
     * @param userId     用户 ID
     * @param type       训练类型（可以为 null）
     * @param startDate  开始日期（可以为 null）
     * @param endDate    结束日期（可以为 null）
     * @param offset     分页偏移量
     * @param limit      每页大小
     * @return 训练数据列表
     */
    List<TrainingDataDO> selectTrainingDataByConditions(
            @Param("userId") Long userId,
            @Param("type") String type,
            @Param("startDate") String startDate,
            @Param("endDate") String endDate,
            @Param("offset") int offset,
            @Param("limit") int limit
    );

    /**
     * 查询用户训练记录的总数（用于分页）
     *
     * @param userId     用户 ID
     * @param type       训练类型（可以为 null）
     * @param startDate  开始日期（可以为 null）
     * @param endDate    结束日期（可以为 null）
     * @return 总数
     */
    int countTrainingDataByConditions(
            @Param("userId") Long userId,
            @Param("type") String type,
            @Param("startDate") String startDate,
            @Param("endDate") String endDate
    );

    // 获取训练数据统计
    Map<String, Object> selectTrainingDataStatistics(
            @Param("userId") Long userId,
            @Param("type") String type,
            @Param("startDate") String startDate,
            @Param("endDate") String endDate
    );
}