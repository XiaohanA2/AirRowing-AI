package com.tongji.airrowing.note.biz.domain.mapper;

import com.tongji.airrowing.note.biz.domain.dataobject.NoteDO;

import java.util.List;
import java.util.Map;

public interface NoteDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(NoteDO record);

    int insertSelective(NoteDO record);

    NoteDO selectByPrimaryKey(Long id);

    int selectCountByNoteId(Long noteId);

    int updateByPrimaryKeySelective(NoteDO record);

    int updateByPrimaryKey(NoteDO record);

    int updateVisibleOnlyMe(NoteDO noteDO);

    int updateIsTop(NoteDO noteDO);

    /**
     * 查询笔记的发布者用户 ID
     * @param noteId
     * @return
     */
    Long selectCreatorIdByNoteId(Long noteId);

    /**
     * 分页查询所有笔记
     *
     * @return 笔记列表
     */
    List<NoteDO> selectAllNotes(Map<String, Object> params);

    /**
     * 获取当前用户的笔记列表
     *
     * @param params 查询参数（包括用户ID、分页、筛选条件等）
     * @return 笔记列表
     */
    List<NoteDO> selectUserNotes(Map<String, Object> params);
}