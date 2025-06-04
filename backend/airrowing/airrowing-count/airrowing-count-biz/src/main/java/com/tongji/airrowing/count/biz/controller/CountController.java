package com.tongji.airrowing.count.biz.controller;

import com.tongji.framework.biz.operationlog.aspect.ApiOperationLog;
import com.tongji.airrowing.count.biz.domain.dataobject.NoteCountDO;
import com.tongji.airrowing.count.biz.domain.dataobject.UserCountDO;
import com.tongji.airrowing.count.biz.service.CountService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/count")
@Slf4j
public class CountController {

    @Resource
    private CountService countService;

    /**
     * 查询指定笔记的计数信息
     *
     * @param request 请求体，包含 noteId
     * @return NoteCountDO 对象，包含计数信息
     */
    @PostMapping("/note")
    @ApiOperationLog(description = "笔记计数信息")
    public NoteCountDO getNoteCount(@RequestBody Map<String, Long> request) {
        Long noteId = request.get("noteId");
        if (noteId == null) {
            throw new IllegalArgumentException("noteId is required");
        }
        return countService.getNoteCountByNoteId(noteId);
    }

    /**
     * 查询用户计数信息
     *
     * @param request 请求体，包含 userId
     * @return UserCountDO 对象，包含计数信息
     */
    @PostMapping("/user")
    @ApiOperationLog(description = "用户计数信息")
    public UserCountDO getUserCount(@RequestBody Map<String, Long> request) {
        Long userId = request.get("userId");
        if (userId == null) {
            throw new IllegalArgumentException("userId is required");
        }
        return countService.getUserCountByUserId(userId);
    }
}
