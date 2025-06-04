package com.tongji.airrowing.count.biz.domain.mapper;

import com.tongji.airrowing.count.biz.domain.dataobject.NoteCountDO;
import org.apache.ibatis.annotations.Param;

public interface NoteCountDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(NoteCountDO record);

    int insertSelective(NoteCountDO record);

    NoteCountDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(NoteCountDO record);

    int updateByPrimaryKey(NoteCountDO record);

    /**
     * 添加笔记计数记录或更新笔记点赞数
     * @param count
     * @param noteId
     * @return
     */
    int insertOrUpdateLikeTotalByNoteId(@Param("count") Integer count, @Param("noteId") Long noteId);

    /**
     * 添加记录或更新笔记收藏数
     * @param count
     * @param noteId
     * @return
     */
    int insertOrUpdateCollectTotalByNoteId(@Param("count") Integer count, @Param("noteId") Long noteId);

    /**
     * 根据笔记ID查询计数信息
     *
     * @param noteId 笔记ID
     * @return NoteCountDO 对象
     */
    NoteCountDO selectByNoteId(@Param("noteId") Long noteId);
}