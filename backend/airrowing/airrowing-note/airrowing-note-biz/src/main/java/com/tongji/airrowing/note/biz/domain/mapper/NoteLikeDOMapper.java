package com.tongji.airrowing.note.biz.domain.mapper;

import com.tongji.airrowing.note.biz.domain.dataobject.NoteLikeDO;
import com.tongji.airrowing.note.biz.model.vo.LikedNoteVO;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

public interface NoteLikeDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(NoteLikeDO record);

    int insertSelective(NoteLikeDO record);

    /**
     * 新增笔记点赞记录，若已存在，则更新笔记点赞记录
     * @param noteLikeDO
     * @return
     */
    int insertOrUpdate(NoteLikeDO noteLikeDO);

    NoteLikeDO selectByPrimaryKey(Long id);

    int selectCountByUserIdAndNoteId(@Param("userId") Long userId, @Param("noteId") Long noteId);

    List<NoteLikeDO> selectByUserId(@Param("userId") Long userId);

    int selectNoteIsLiked(@Param("userId") Long userId, @Param("noteId") Long noteId);

    List<NoteLikeDO> selectLikedByUserIdAndLimit(@Param("userId") Long userId, @Param("limit")  int limit);

    int updateByPrimaryKeySelective(NoteLikeDO record);

    int updateByPrimaryKey(NoteLikeDO record);

    /**
     * 取消点赞
     * @param noteLikeDO
     * @return
     */
    int update2UnlikeByUserIdAndNoteId(NoteLikeDO noteLikeDO);

    /**
     * 分页查询用户点赞过的笔记，包含笔记的标题和创建者ID
     *
     * @param params 查询参数，包括 userId、offset、limit
     * @return 点赞的笔记列表
     */
    List<LikedNoteVO> selectLikedNotesByUserIdWithPagination(Map<String, Object> params);
}