package com.tongji.airrowing.club.biz.controller;

import com.tongji.airrowing.club.biz.model.vo.*;
import com.tongji.airrowing.club.biz.service.TrainingDataService;
import com.tongji.framework.biz.context.holder.LoginUserContextHolder;
import com.tongji.framework.common.response.Response;
import com.tongji.framework.biz.operationlog.aspect.ApiOperationLog;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 训练数据控制器
 */
@RestController
@RequestMapping("/training-data")
@Slf4j
public class TrainingDataController {

    @Resource
    private TrainingDataService trainingDataService;

    /**
     * 提交用户训练数据
     *
     * @param trainingDataVO 训练数据请求体
     * @return 操作结果
     */
    @PostMapping("/submit")
    @ApiOperationLog(description = "用户提交训练数据")
    public Response<Long> submitTrainingData(@Validated @RequestBody TrainingDataVO trainingDataVO) {
        // 调用服务层保存数据并返回响应
        return trainingDataService.saveTrainingData(trainingDataVO);
    }

    @PostMapping("/query")
    @ApiOperationLog(description = "查询用户训练记录")
    public Response<Map<String, Object>> queryTrainingData(
            @RequestBody TrainingDataQueryVO queryVO) {  // 使用 @RequestBody 接收整个 VO 对象

        // 获取当前用户的 userId（从上下文中获取，不需要从请求中传递）
        Long currentUserId = LoginUserContextHolder.getUserId();
        queryVO.setUserId(currentUserId);  // 设置 VO 中的 userId

        // 调用 service 层方法，传入 VO 中的查询条件
        Response<Map<String, Object>> response = trainingDataService.queryTrainingData(
                queryVO.getPage(), queryVO.getSize(), queryVO.getType(), queryVO.getStartDate(), queryVO.getEndDate());

        // 从 Response 中获取实际的 Map 数据
        Map<String, Object> result = response.getData();

        // 返回成功的响应
        return Response.success(result);
    }

    @PostMapping("/update")
    @ApiOperationLog(description = "更新训练记录")
    public Response<String> updateTrainingData(@RequestBody TrainingDataUpdateVO updateVO) {
        // 调用 Service 层的更新方法
        return trainingDataService.updateTrainingData(updateVO);
    }


    @PostMapping("/delete")
    @ApiOperationLog(description = "删除用户训练记录")
    public Response<String> deleteTrainingData(@RequestBody TrainingDataDeleteRequest request) {
        Long currentUserId = LoginUserContextHolder.getUserId();  // 获取当前登录用户的ID
        return trainingDataService.deleteTrainingData(request.getId(), currentUserId);
    }

    @PostMapping("/statistics")
    @ApiOperationLog(description = "获取用户训练数据统计")
    public Response<Map<String, Object>> getTrainingDataStatistics(@RequestBody TrainingDataStatisticsRequest request) {
        Long currentUserId = LoginUserContextHolder.getUserId();  // 获取当前登录用户的ID
        return trainingDataService.getTrainingDataStatistics(currentUserId, request);
    }





}
