package com.tongji.airrowing.club.biz.service;

import cn.hutool.db.PageResult;
import com.tongji.airrowing.club.biz.domain.dataobject.TrainingDataDO;
import com.tongji.airrowing.club.biz.model.dto.TrainingDataDTO;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataStatisticsRequest;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataUpdateVO;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataVO;
import com.tongji.airrowing.club.biz.model.vo.TrainingRecordVO;
import com.tongji.framework.common.response.Response;

import java.util.Map;

public interface TrainingDataService {
    /**
     * 保存训练数据
     *
     * @param trainingDataVO 训练数据对象
     * @return 操作结果
     */
    Response<Long> saveTrainingData(TrainingDataVO trainingDataVO);

    /**
     * 查询用户训练记录，支持分页和筛选
     *
     * @param page      页码
     * @param size      每页大小
     * @param type      训练类型（可选）
     * @param startDate 开始日期（可选）
     * @param endDate   结束日期（可选）
     * @return 查询结果
     */
    Response<Map<String, Object>> queryTrainingData(int page, int size, String type, String startDate, String endDate);

    Response<String> updateTrainingData(TrainingDataUpdateVO updateVO);

    Response<String> deleteTrainingData(Long id, Long userId);

    Response<Map<String, Object>> getTrainingDataStatistics(Long userId, TrainingDataStatisticsRequest request);

}
