package com.tongji.airrowing.note.biz.controller;

import com.tongji.framework.biz.operationlog.aspect.ApiOperationLog;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.note.biz.model.vo.*;
import com.tongji.airrowing.note.biz.service.NoteService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/note")
@Slf4j
public class NoteController {

    @Resource
    private NoteService noteService;

    @PostMapping(value = "/publish")
    @ApiOperationLog(description = "笔记发布")
    public Response<?> publishNote(@Validated @RequestBody PublishNoteReqVO publishNoteReqVO) {
        return noteService.publishNote(publishNoteReqVO);
    }

    @PostMapping(value = "/detail")
    @ApiOperationLog(description = "笔记详情")
    public Response<FindNoteDetailRspVO> findNoteDetail(@Validated @RequestBody FindNoteDetailReqVO findNoteDetailReqVO) {
        return noteService.findNoteDetail(findNoteDetailReqVO);
    }

    @PostMapping(value = "/update")
    @ApiOperationLog(description = "笔记修改")
    public Response<?> updateNote(@Validated @RequestBody UpdateNoteReqVO updateNoteReqVO) {
        return noteService.updateNote(updateNoteReqVO);
    }


    @PostMapping(value = "/delete")
    @ApiOperationLog(description = "删除笔记")
    public Response<?> deleteNote(@Validated @RequestBody DeleteNoteReqVO deleteNoteReqVO) {
        return noteService.deleteNote(deleteNoteReqVO);
    }

    @PostMapping(value = "/visible/onlyme")
    @ApiOperationLog(description = "笔记仅对自己可见")
    public Response<?> visibleOnlyMe(@Validated @RequestBody UpdateNoteVisibleOnlyMeReqVO updateNoteVisibleOnlyMeReqVO) {
        return noteService.visibleOnlyMe(updateNoteVisibleOnlyMeReqVO);
    }

    @PostMapping(value = "/top")
    @ApiOperationLog(description = "置顶/取消置顶笔记")
    public Response<?> topNote(@Validated @RequestBody TopNoteReqVO topNoteReqVO) {
        return noteService.topNote(topNoteReqVO);
    }

    @PostMapping(value = "/like")
    @ApiOperationLog(description = "点赞笔记")
    public Response<?> likeNote(@Validated @RequestBody LikeNoteReqVO likeNoteReqVO) {
        return noteService.likeNote(likeNoteReqVO);
    }

    @PostMapping(value = "/unlike")
    @ApiOperationLog(description = "取消点赞笔记")
    public Response<?> unlikeNote(@Validated @RequestBody UnlikeNoteReqVO unlikeNoteReqVO) {
        return noteService.unlikeNote(unlikeNoteReqVO);
    }

    @PostMapping(value = "/collect")
    @ApiOperationLog(description = "收藏笔记")
    public Response<?> collectNote(@Validated @RequestBody CollectNoteReqVO collectNoteReqVO) {
        return noteService.collectNote(collectNoteReqVO);
    }

    @PostMapping(value = "/uncollect")
    @ApiOperationLog(description = "取消收藏笔记")
    public Response<?> unCollectNote(@Validated @RequestBody UnCollectNoteReqVO unCollectNoteReqVO) {
        return noteService.unCollectNote(unCollectNoteReqVO);
    }

    /**
     * 分页查询所有笔记列表（POST 方法）
     */
    @PostMapping(value = "/list")
    public Response<?> getNoteList(@RequestBody NoteListReqVO noteListReqVO) {
        // 通过请求体获取 page 和 size 参数
        return noteService.getNoteList(noteListReqVO.getPage(), noteListReqVO.getSize());
    }

    /**
     * 获取当前用户的笔记列表
     *
     * @param request 请求体参数（包括用户ID、分页、筛选条件等）
     * @return 当前用户的笔记列表
     */
    @PostMapping("/UserNoteList")
    public Response<?> getUserNoteList(@RequestBody GetUserNotesRequest request) {
        return noteService.getUserNotes(
                request.getUserId(),
                request.getPage(),
                request.getSize(),
                request.getIsTop(),
                request.getVisible(),
                request.getStatus()
        );
    }

    @PostMapping("/liked/list")
    @ApiOperationLog(description = "分页查询用户点赞过的笔记")
    public Response<List<LikedNoteVO>> getUserLikedNotes(@Validated @RequestBody GetUserLikedNotesRequest request) {
        return noteService.getUserLikedNotes(
                request.getUserId(),
                request.getPage(),
                request.getSize()
        );
    }

    @PostMapping("/collected/list")
    @ApiOperationLog(description = "分页查询用户收藏的笔记")
    public Response<List<CollectedNoteVO>> getUserCollectedNotes(@Validated @RequestBody GetUserCollectedNotesRequest request) {
        return noteService.getUserCollectedNotes(
                request.getUserId(),
                request.getPage(),
                request.getSize()
        );
    }
}
