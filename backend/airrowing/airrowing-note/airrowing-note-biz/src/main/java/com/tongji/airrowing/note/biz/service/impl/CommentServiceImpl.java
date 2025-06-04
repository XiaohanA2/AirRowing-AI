package com.tongji.airrowing.note.biz.service.impl;

import com.alibaba.nacos.shaded.com.google.common.base.Preconditions;
import com.tongji.framework.biz.context.holder.LoginUserContextHolder;
import com.tongji.framework.common.exception.BizException;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.note.biz.domain.dataobject.CommentDO;
import com.tongji.airrowing.note.biz.domain.mapper.CommentDOMapper;
import com.tongji.airrowing.note.biz.enums.ResponseCodeEnum;
import com.tongji.airrowing.note.biz.model.vo.CommentReqVO;
import com.tongji.airrowing.note.biz.model.vo.CommentRspVO;
import com.tongji.airrowing.note.biz.model.vo.CommentWithUserVO;
import com.tongji.airrowing.note.biz.rpc.DistributedIdGeneratorRpcService;
import com.tongji.airrowing.note.biz.service.CommentService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Slf4j
public class CommentServiceImpl implements CommentService {

    @Resource
    private CommentDOMapper commentDOMapper;

    @Resource
    private DistributedIdGeneratorRpcService distributedIdGeneratorRpcService;

    //增加评论
    @Override
    public Response<?> addComment(CommentReqVO commentReqVO) {
        // 从上下文中获取当前用户ID
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 参数校验
        if (commentReqVO.getContent().trim().isEmpty()) {
            throw new BizException(ResponseCodeEnum.COMMENT_BLANK);
        }

        // 校验 noteId 是否存在
        boolean noteExists = commentDOMapper.noteExists(commentReqVO.getNoteId());
        if (!noteExists) {
            throw new BizException(ResponseCodeEnum.NOTE_NOT_FOUND);
        }

        // 如果 parentId 不为空，校验 parentId 是否存在
        if (commentReqVO.getParentId() != null) {
            boolean parentExists = commentDOMapper.commentExists(commentReqVO.getParentId());
            if (!parentExists) {
                throw new BizException(ResponseCodeEnum.COMMENT_NOT_FOUND);
            }
        }

        // RPC: 调用分布式 ID 生成服务，生成评论 ID
        String snowflakeIdId = distributedIdGeneratorRpcService.getSnowflakeId();

        // 构建 CommentDO 对象
        CommentDO commentDO = CommentDO.builder()
                .id(Long.valueOf(snowflakeIdId))
                .noteId(commentReqVO.getNoteId())
                .userId(currentUserId) // 使用当前用户ID
                .parentId(commentReqVO.getParentId())
                .content(commentReqVO.getContent().trim())
                .status((byte) 0) // 默认状态为正常
                .createTime(LocalDateTime.now())
                .updateTime(LocalDateTime.now())
                .build();

        // 插入数据库
        int result = commentDOMapper.insertSelective(commentDO);

        if (result > 0) {
            log.info("评论添加成功: {}", commentDO);
            return Response.success();
        } else {
            log.error("评论添加失败: {}", commentDO);
            throw new BizException(ResponseCodeEnum.COMMENT_ERROR);
        }
    }

    //删除评论
    @Override
    public Response<?> deleteComment(Long commentId) {
        // 从上下文中获取当前用户ID
        Long currentUserId = LoginUserContextHolder.getUserId();

        // 查询评论信息
        CommentDO comment = commentDOMapper.selectByPrimaryKey(commentId);
        if (comment == null || comment.getStatus() != 0) {
            throw new BizException(ResponseCodeEnum.COMMENT_NOT_FOUND);
        }

        // 查询笔记发布者ID
        Long noteCreatorId = commentDOMapper.selectNoteCreatorIdByNoteId(comment.getNoteId());
        if (noteCreatorId == null) {
            throw new BizException(ResponseCodeEnum.NOTE_NOT_FOUND);
        }

        // 权限校验：当前用户必须是评论发布者或笔记发布者
        if (!currentUserId.equals(comment.getUserId()) && !currentUserId.equals(noteCreatorId)) {
            throw new BizException(ResponseCodeEnum.PERMISSION_DENIED);
        }

        // 删除评论及其所有子评论（逻辑删除）
        int rowsAffected = commentDOMapper.deleteCommentAndReplies(commentId);

        if (rowsAffected > 0) {
            log.info("评论删除成功，评论ID: {}, 删除用户ID: {}", commentId, currentUserId);
            return Response.success();
        } else {
            log.error("评论删除失败，评论ID: {}", commentId);
            throw new BizException(ResponseCodeEnum.COMMENT_ERROR);
        }
    }

    //查看一级评论
    @Override
    public Response<List<CommentWithUserVO>> getCommentsByNoteId(Long noteId, int page, int size) {
        // 参数校验
        Preconditions.checkArgument(noteId != null && noteId > 0, "笔记ID不能为空且必须大于0");
        Preconditions.checkArgument(page > 0, "页码必须大于0");
        Preconditions.checkArgument(size > 0, "每页大小必须大于0");

        // 计算分页参数
        int offset = (page - 1) * size;

        // 查询评论及用户信息
        List<CommentWithUserVO> commentWithUserVOList =
                commentDOMapper.selectCommentsWithUserByNoteId(noteId, offset, size);

        // 返回结果
        return Response.success(commentWithUserVOList);
    }

    //查看二级评论
    @Override
    public Response<List<CommentRspVO>> getRepliesByCommentId(Long commentId) {
        // 参数校验
        Preconditions.checkArgument(commentId != null && commentId > 0, "父评论ID不能为空且必须大于0");

        // 查询子评论列表（包含用户信息）
        List<CommentRspVO> replies = commentDOMapper.selectRepliesWithUserByCommentId(commentId)
                .stream()
                .map(commentDO -> {
                    CommentRspVO commentRspVO = new CommentRspVO();
                    commentRspVO.setId(commentDO.getId());
                    commentRspVO.setNoteId(commentDO.getNoteId());
                    commentRspVO.setUserId(commentDO.getUserId());
                    commentRspVO.setUserName(commentDO.getUserName());
                    commentRspVO.setUserAvatar(commentDO.getUserAvatar());
                    commentRspVO.setContent(commentDO.getContent());
                    commentRspVO.setCreateTime(commentDO.getCreateTime());
                    commentRspVO.setUpdateTime(commentDO.getUpdateTime());
                    commentRspVO.setParentId(commentDO.getParentId());

                    // 递归查询子评论
                    List<CommentRspVO> childReplies = getRepliesByCommentId(commentDO.getId()).getData();
                    commentRspVO.setReplies(childReplies);

                    return commentRspVO;
                })
                .collect(Collectors.toList());

        return Response.success(replies);
    }

    //统计某笔记的评论数
    @Override
    public Response<Integer> countCommentsByNoteId(Long noteId) {
        // 参数校验
        Preconditions.checkArgument(noteId != null && noteId > 0, "笔记ID不能为空且必须大于0");

        // 查询评论总数
        int count = commentDOMapper.countCommentsByNoteId(noteId);

        return Response.success(count);
    }

}
