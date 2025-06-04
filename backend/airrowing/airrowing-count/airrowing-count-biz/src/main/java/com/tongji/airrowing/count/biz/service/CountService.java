package com.tongji.airrowing.count.biz.service;

import com.tongji.airrowing.count.biz.domain.dataobject.NoteCountDO;
import com.tongji.airrowing.count.biz.domain.dataobject.UserCountDO;

public interface CountService {

    /**
     * 查询指定笔记的所有计数信息
     *
     * @param noteId 笔记ID
     * @return NoteCountDO 对象，包含该笔记的计数信息
     */
    NoteCountDO getNoteCountByNoteId(Long noteId);

    /**
     * 查询指定用户的计数信息
     *
     * @param userId 用户ID
     * @return UserCountDO 对象，包含该用户的计数信息
     */
    UserCountDO getUserCountByUserId(Long userId);

}
