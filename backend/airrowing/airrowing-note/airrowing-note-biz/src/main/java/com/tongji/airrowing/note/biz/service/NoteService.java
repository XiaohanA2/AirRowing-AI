package com.tongji.airrowing.note.biz.service;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.note.biz.model.vo.*;

import java.util.List;

public interface NoteService {

    /**
     * 笔记发布
     * @param publishNoteReqVO
     * @return
     */
    Response<?> publishNote(PublishNoteReqVO publishNoteReqVO);

    /**
     * 笔记详情
     * @param findNoteDetailReqVO
     * @return
     */
    Response<FindNoteDetailRspVO> findNoteDetail(FindNoteDetailReqVO findNoteDetailReqVO);

    /**
     * 笔记更新
     * @param updateNoteReqVO
     * @return
     */
    Response<?> updateNote(UpdateNoteReqVO updateNoteReqVO);

    /**
     * 删除本地笔记缓存
     * @param noteId
     */
    void deleteNoteLocalCache(Long noteId);

    /**
     * 删除笔记
     * @param deleteNoteReqVO
     * @return
     */
    Response<?> deleteNote(DeleteNoteReqVO deleteNoteReqVO);

    /**
     * 笔记仅对自己可见
     * @param updateNoteVisibleOnlyMeReqVO
     * @return
     */
    Response<?> visibleOnlyMe(UpdateNoteVisibleOnlyMeReqVO updateNoteVisibleOnlyMeReqVO);

    /**
     * 笔记置顶 / 取消置顶
     * @param topNoteReqVO
     * @return
     */
    Response<?> topNote(TopNoteReqVO topNoteReqVO);

    /**
     * 点赞笔记
     * @param likeNoteReqVO
     * @return
     */
    Response<?> likeNote(LikeNoteReqVO likeNoteReqVO);

    /**
     * 取消点赞笔记
     * @param unlikeNoteReqVO
     * @return
     */
    Response<?> unlikeNote(UnlikeNoteReqVO unlikeNoteReqVO);

    /**
     * 收藏笔记
     * @param collectNoteReqVO
     * @return
     */
    Response<?> collectNote(CollectNoteReqVO collectNoteReqVO);

    /**
     * 取消收藏笔记
     * @param unCollectNoteReqVO
     * @return
     */
    Response<?> unCollectNote(UnCollectNoteReqVO unCollectNoteReqVO);

    /**
     * 分页查询所有笔记列表
     */
    Response<?> getNoteList(int page, int size);

    Response<?> getUserNotes(Long userId, int page, int size, Integer isTop, Integer visible, Integer status);

    /**
     * 分页查询用户点赞过的笔记
     *
     * @param userId 用户ID
     * @param page   页码，从1开始
     * @param size   每页大小
     * @return 用户点赞的笔记列表
     */
    Response<List<LikedNoteVO>> getUserLikedNotes(Long userId, int page, int size);
    /**
     * 分页查询用户收藏的笔记
     *
     * @param userId 用户ID
     * @param page   页码，从1开始
     * @param size   每页大小
     * @return 用户收藏的笔记列表
     */
    Response<List<CollectedNoteVO>> getUserCollectedNotes(Long userId, int page, int size);
}
