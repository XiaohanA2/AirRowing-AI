package com.tongji.airrowing.club.biz.service.impl;

import cn.hutool.db.PageResult;
import com.tongji.airrowing.club.biz.domain.dataobject.TrainingDataDO;
import com.tongji.airrowing.club.biz.domain.mapper.TrainingDataDOMapper;
import com.tongji.airrowing.club.biz.enums.ResponseCodeEnum;
import com.tongji.airrowing.club.biz.model.dto.TrainingDataDTO;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataStatisticsRequest;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataUpdateVO;
import com.tongji.airrowing.club.biz.model.vo.TrainingDataVO;
import com.tongji.airrowing.club.biz.model.vo.TrainingRecordVO;
import com.tongji.airrowing.club.biz.rpc.DistributedIdGeneratorRpcService;
import com.tongji.airrowing.club.biz.service.TrainingDataService;
import com.tongji.framework.common.exception.BizException;
import com.tongji.framework.common.response.Response;
import com.tongji.framework.biz.context.holder.LoginUserContextHolder;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.sql.Date;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@Slf4j
public class TrainingDataServiceImpl implements TrainingDataService {

    @Resource
    private TrainingDataDOMapper trainingDataDOMapper;

    @Resource
    private DistributedIdGeneratorRpcService distributedIdGeneratorRpcService;

    /**
     * 保存用户训练数据
     *
     * @param trainingDataVO 训练数据视图对象
     * @return 操作结果
     */
    @Override
    @Transactional
    public Response<Long> saveTrainingData(TrainingDataVO trainingDataVO) {
        // 获取当前登录用户的 ID
        Long userId = LoginUserContextHolder.getUserId();
        if (userId == null) {
            throw new BizException(ResponseCodeEnum.ID_NOT_VALID);
        }

        // 校验必要字段是否为空
        if (trainingDataVO.getTrainingDate() == null || trainingDataVO.getDuration() == null) {
            throw new BizException(ResponseCodeEnum.DATA_NOT_VALID);
        }

        // 创建 TrainingDataDTO 对象并设置值
        TrainingDataDTO trainingDataDTO = new TrainingDataDTO();
        trainingDataDTO.setUserId(userId);
        trainingDataDTO.setTrainingDate(trainingDataVO.getTrainingDate());
        trainingDataDTO.setDuration(trainingDataVO.getDuration());
        trainingDataDTO.setDistance(trainingDataVO.getDistance());
        trainingDataDTO.setCalories(trainingDataVO.getCalories());
        trainingDataDTO.setStatus(trainingDataVO.getStatus());
        trainingDataDTO.setType(trainingDataVO.getType());

        // 创建 TrainingDataDO 对象并插入数据库
        // 调用分布式 ID 生成服务，生成训练数据 ID
        String trainId = distributedIdGeneratorRpcService.getSnowflakeId();
        TrainingDataDO trainingDataDO = TrainingDataDO.builder()
                .id(Long.valueOf(trainId))
                .userId(trainingDataDTO.getUserId())
                .trainingDate(trainingDataDTO.getTrainingDate())
                .duration(trainingDataDTO.getDuration())
                .distance(trainingDataDTO.getDistance())
                .calories(trainingDataDTO.getCalories())
                .status(trainingDataDTO.getStatus())
                .type(trainingDataDTO.getType())
                .createTime(LocalDateTime.now())
                .updateTime(LocalDateTime.now())
                .build();

        try {
            // 插入数据到数据库
            trainingDataDOMapper.insertSelective(trainingDataDO);
        } catch (Exception e) {
            log.error("==> Failed to save training data", e);
            throw new BizException(ResponseCodeEnum.DATA_NOT_SAVED);
        }

        // 返回保存成功后的训练数据 ID
        return Response.success(trainingDataDO.getId());
    }

    @Override
    @Transactional
    public Response<Map<String, Object>> queryTrainingData(int page, int size, String type, String startDate, String endDate) {
        // 获取当前用户的 userId
        Long userId = LoginUserContextHolder.getUserId();

        // 计算分页偏移量
        int offset = (page - 1) * size;

        // 获取用户的训练数据
        List<TrainingDataDO> trainingDataList = trainingDataDOMapper.selectTrainingDataByConditions(
                userId, type, startDate, endDate, offset, size);

        // 统计总条数
        int total = trainingDataDOMapper.countTrainingDataByConditions(userId, type, startDate, endDate);

        // 转换成新的 VO 对象
        List<TrainingRecordVO> result = trainingDataList.stream()
                .map(trainingData -> new TrainingRecordVO(
                        trainingData.getId(),
                        trainingData.getTrainingDate(),
                        trainingData.getDuration(),
                        trainingData.getDistance(),
                        trainingData.getCalories(),
                        trainingData.getType()
                ))
                .collect(Collectors.toList());

        // 构建分页数据返回
        Map<String, Object> pageResult = new HashMap<>();
        pageResult.put("list", result);
        pageResult.put("total", total);
        pageResult.put("page", page);
        pageResult.put("size", size);

        // 返回成功响应
        return Response.success(pageResult);
    }

    @Override
    @Transactional
    public Response<String> updateTrainingData(TrainingDataUpdateVO updateVO) {
        // 获取当前用户的 userId
        Long userId = LoginUserContextHolder.getUserId();

        // 根据 ID 查询现有的训练记录
        TrainingDataDO existingTrainingData = trainingDataDOMapper.selectByPrimaryKey(updateVO.getId());

        // 检查记录是否存在
        if (existingTrainingData == null) {
            throw new BizException(ResponseCodeEnum.PARAM_NOT_VALID);
        }

        // 确保更新的记录属于当前用户
        if (!existingTrainingData.getUserId().equals(userId)) {
            throw new BizException(ResponseCodeEnum.PARAM_NOT_VALID);
        }

        // 根据 VO 中的字段更新训练记录
        if (updateVO.getDuration() != null) {
            existingTrainingData.setDuration(updateVO.getDuration());
        }
        if (updateVO.getDistance() != null) {
            existingTrainingData.setDistance(updateVO.getDistance());
        }
        if (updateVO.getCalories() != null) {
            existingTrainingData.setCalories(updateVO.getCalories());
        }
        if (updateVO.getType() != null) {
            existingTrainingData.setType(updateVO.getType());
        }
        if (updateVO.getTrainingDate() != null) {
            // 将字符串日期转换为 LocalDateTime 类型
            LocalDateTime trainingDateTime = LocalDate.parse(updateVO.getTrainingDate()).atStartOfDay();
            existingTrainingData.setTrainingDate(trainingDateTime);
        }

        // 执行更新操作
        int rowsAffected = trainingDataDOMapper.updateByPrimaryKeySelective(existingTrainingData);

        // 判断更新是否成功
        if (rowsAffected > 0) {
            return Response.success("训练记录更新成功");
        } else {
            throw new BizException(ResponseCodeEnum.PARAM_NOT_VALID);
        }
    }

    @Override
    @Transactional
    public Response<String> deleteTrainingData(Long id, Long userId) {
        // 根据 ID 查询现有的训练记录
        TrainingDataDO existingTrainingData = trainingDataDOMapper.selectByPrimaryKey(id);

        // 检查记录是否存在
        if (existingTrainingData == null) {
            throw new BizException(ResponseCodeEnum.ACTIVITY_NOT_FOUND);
        }

        // 确保删除的记录属于当前用户
        if (!existingTrainingData.getUserId().equals(userId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 执行删除操作
        int rowsAffected = trainingDataDOMapper.deleteByPrimaryKey(id);

        // 判断删除是否成功
        if (rowsAffected > 0) {
            return Response.success("训练记录删除成功");
        } else {
            throw new BizException(ResponseCodeEnum.PARAM_NOT_VALID);
        }
    }

    @Override
    @Transactional
    public Response<Map<String, Object>> getTrainingDataStatistics(Long userId, TrainingDataStatisticsRequest request) {
        // 计算统计条件的偏移量和筛选条件
        String type = request.getType();
        String startDate = request.getStartDate();
        String endDate = request.getEndDate();

        // 获取统计数据
        Map<String, Object> statistics = trainingDataDOMapper.selectTrainingDataStatistics(userId, type, startDate, endDate);

        if (statistics == null || statistics.isEmpty()){
            throw new BizException(ResponseCodeEnum.DATA_NOT_FIND);
        }

        return Response.success(statistics);
    }


}


