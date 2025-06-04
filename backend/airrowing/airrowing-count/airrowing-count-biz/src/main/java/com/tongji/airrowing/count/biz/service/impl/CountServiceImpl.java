package com.tongji.airrowing.count.biz.service.impl;

import com.tongji.framework.common.exception.BizException;
import com.tongji.airrowing.count.biz.domain.dataobject.NoteCountDO;
import com.tongji.airrowing.count.biz.domain.dataobject.UserCountDO;
import com.tongji.airrowing.count.biz.domain.mapper.NoteCountDOMapper;
import com.tongji.airrowing.count.biz.domain.mapper.UserCountDOMapper;
import com.tongji.airrowing.count.biz.enums.ResponseCodeEnum;
import com.tongji.airrowing.count.biz.service.CountService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class CountServiceImpl implements CountService {

    @Resource
    private NoteCountDOMapper noteCountDOMapper;

    @Resource
    private UserCountDOMapper userCountDOMapper;
    @Override
    public NoteCountDO getNoteCountByNoteId(Long noteId) {
        NoteCountDO noteCountDO = noteCountDOMapper.selectByNoteId(noteId);
        if (noteCountDO == null) {
            // 当记录不存在时，抛出业务异常
            throw new BizException(ResponseCodeEnum.NOTE_NOT_FOUND);
        }
        return noteCountDO;
    }

    @Override
    public UserCountDO getUserCountByUserId(Long userId) {
        // 使用 Optional 判断 userCountDO 是否为 null，不存在时抛出 BizException
        return Optional.ofNullable(userCountDOMapper.selectByUserId(userId))
                .orElseThrow(() -> new BizException(ResponseCodeEnum.NOTE_NOT_FOUND));
    }
}