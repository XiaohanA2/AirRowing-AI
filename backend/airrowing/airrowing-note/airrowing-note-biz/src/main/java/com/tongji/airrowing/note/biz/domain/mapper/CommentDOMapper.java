package com.tongji.airrowing.note.biz.domain.mapper;

import com.tongji.airrowing.note.biz.domain.dataobject.CommentDO;
import com.tongji.airrowing.note.biz.model.vo.CommentRspVO;
import com.tongji.airrowing.note.biz.model.vo.CommentWithUserVO;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface CommentDOMapper {
    int deleteByPrimaryKey(Long id);

    int insert(CommentDO record);

    int insertSelective(CommentDO record);

    CommentDO selectByPrimaryKey(Long id);

    int updateByPrimaryKeySelective(CommentDO record);

    int updateByPrimaryKeyWithBLOBs(CommentDO record);

    int updateByPrimaryKey(CommentDO record);

    /**
     * 校验笔记是否存在
     * @param noteId 笔记ID
     * @return 是否存在
     */
    boolean noteExists(@Param("noteId") Long noteId);

    /**
     * 校验评论是否存在
     * @param commentId 评论ID
     * @return 是否存在
     */
    boolean commentExists(@Param("commentId") Long commentId);

    /**
     * 查询笔记的创建者ID
     *
     * @param noteId 笔记ID
     * @return 笔记发布者ID
     */
    Long selectNoteCreatorIdByNoteId(@Param("noteId") Long noteId);

    /**
     * 删除评论及其所有子评论（逻辑删除）
     *
     * @param commentId 评论ID
     * @return 受影响的行数
     */
    int deleteCommentAndReplies(@Param("commentId") Long commentId);

    /**
     * 分页查询某篇笔记的一级评论，同时关联查询用户信息
     *
     * @param noteId 笔记ID
     * @param offset 偏移量
     * @param limit  每页大小
     * @return 评论列表（包含用户信息）
     */
    List<CommentWithUserVO> selectCommentsWithUserByNoteId(@Param("noteId") Long noteId,
                                                           @Param("offset") int offset,
                                                           @Param("limit") int limit);

    /**
     * 查询某条评论的所有子评论，同时关联用户信息
     *
     * @param parentId 父评论ID
     * @return 子评论列表
     */
    List<CommentRspVO> selectRepliesWithUserByCommentId(@Param("parentId") Long parentId);

    /**
     * 统计某篇笔记的评论总数
     *
     * @param noteId 笔记ID
     * @return 评论总数
     */
    int countCommentsByNoteId(@Param("noteId") Long noteId);
}