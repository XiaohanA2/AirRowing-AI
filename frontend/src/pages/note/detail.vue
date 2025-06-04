<template>
    <div class="page-container">
        <el-card class="content-card" shadow="never" v-if="note">
            <!-- 文章头部 -->
            <div class="article-header">
                <h1 class="article-title">{{ note.title }}</h1>
                <div class="article-meta">
                    <div class="author-info" @click="goToUserDetail(note.creatorId)">
                        <img :src="note.avatar" alt="Creator Avatar" class="author-avatar" />
                        <div class="author-details">
                            <span class="author-name">{{ note.creatorName }}</span>
                            <span class="publish-time">{{ note.updateTime }}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 图片轮播 -->
            <div class="image-gallery" v-if="note.imgUris && note.imgUris.length">
                <el-carousel :interval="4000" type="card" height="400px">
                    <el-carousel-item v-for="(img, index) in note.imgUris" :key="index">
                        <div class="carousel-image-wrapper">
                            <img :src="img" :alt="`Image ${index + 1}`" class="carousel-image" />
                        </div>
                    </el-carousel-item>
                </el-carousel>
            </div>

            <!-- 文章内容 -->
            <div class="article-content">
                <p>{{ note.content }}</p>
            </div>

            <!-- 评论区 -->
            <div class="comments-section" id="comments">
                <div class="comments-header">
                    <h3 class="section-title">评论 ({{ commentCount }})</h3>
                </div>

                <!-- 评论输入框 -->
                <div class="comment-form">
                    <el-input 
                        v-model="newComment.content"
                        type="textarea"
                        :rows="3"
                        placeholder="写下你的评论..."
                        :maxlength="500"
                        show-word-limit
                    />
                    <div class="comment-tools">
                        <!-- 表情选择器改为下拉菜单 -->
                        <el-dropdown @command="insertEmoji" trigger="click">
                            <el-button type="text" class="tool-btn">
                                😊 表情
                            </el-button>
                            <template #dropdown>
                                <el-dropdown-menu class="emoji-grid-menu">
                                    <div class="emoji-grid-container">
                                        <el-dropdown-item 
                                            v-for="emoji in commonEmojis" 
                                            :key="emoji"
                                            :command="emoji"
                                            class="emoji-grid-item">
                                            {{ emoji }}
                                        </el-dropdown-item>
                                    </div>
                                </el-dropdown-menu>
                            </template>
                        </el-dropdown>

                        <!-- 快捷回复保持不变 -->
                        <el-dropdown @command="insertQuickReply">
                            <el-button type="text" class="tool-btn">
                                <el-icon><ChatLineSquare /></el-icon>
                                快捷回复
                            </el-button>
                            <template #dropdown>
                                <el-dropdown-menu>
                                    <el-dropdown-item 
                                        v-for="(reply, index) in quickReplies"
                                        :key="index"
                                        :command="reply">
                                        {{ reply }}
                                    </el-dropdown-item>
                                </el-dropdown-menu>
                            </template>
                        </el-dropdown>
                    </div>
                    <div class="form-actions">
                        <el-button type="primary" @click="submitComment" :loading="loading.submit">
                            发布评论
                        </el-button>
                    </div>
                </div>

                <!-- 评论列表 -->
                <div class="comments-list" v-loading="loading.comments">
                    <template v-if="comments.length">
                        <div v-for="comment in comments" 
                             :key="comment.commentId" 
                             class="comment-item">
                            <div class="comment-main">
                                <div class="comment-user">
                                    <img :src="comment.userAvatar" 
                                         :alt="comment.userName" 
                                         class="user-avatar"
                                         @click="goToUserDetail(comment.userId)" />
                                    <div class="user-info">
                                        <span class="user-name">{{ comment.userName }}</span>
                                        <span class="comment-time">{{ comment.createTime }}</span>
                                    </div>
                                </div>
                                <div class="comment-content">{{ comment.content }}</div>
                                <div class="comment-actions">
                                    <el-button type="text" @click="toggleCommentReply(comment)">
                                        回复
                                    </el-button>
                                    <el-button 
                                        v-if="comment.replies?.length"
                                        type="text"
                                        @click="toggleReplies(comment.commentId)">
                                        {{ isRepliesCollapsed[comment.commentId] ? '展开回复' : '收起回复' }}
                                        ({{ comment.replies.length }})
                                    </el-button>
                                </div>

                                <!-- 回复列表 -->
                                <div v-if="!isRepliesCollapsed[comment.commentId] && comment.replies?.length"
                                     class="replies-list">
                                    <div v-for="reply in comment.replies"
                                         :key="reply.id"
                                         class="reply-item">
                                        <div class="reply-content">
                                            <span class="reply-user" @click="goToUserDetail(reply.userId)">{{ reply.userName }}</span>
                                            <template v-if="reply.replyToUserId">
                                                <span class="reply-arrow">回复</span>
                                                <span class="reply-to-user" @click="goToUserDetail(reply.replyToUserId)">@{{ reply.replyToUserName }}</span>
                                            </template>
                                            <span class="reply-text">：{{ reply.content }}</span>
                                        </div>
                                        <div class="reply-footer">
                                            <span class="reply-time">{{ reply.createTime }}</span>
                                            <div class="reply-actions">
                                                <el-button type="text" size="small" @click="toggleNestedReply(comment.commentId, reply)">
                                                    回复
                                                </el-button>
                                                <el-button 
                                                    v-if="isCurrentUser(reply.userId)"
                                                    type="text" 
                                                    size="small" 
                                                    @click="deleteReply(comment.commentId, reply.id)">
                                                    删除
                                                </el-button>
                                            </div>
                                        </div>
                                        <!-- 嵌套回复的输入框 -->
                                        <div v-if="replyInputs[`${comment.commentId}-${reply.id}`]" class="nested-reply-form">
                                            <el-input 
                                                v-model="replyContent[`${comment.commentId}-${reply.id}`]"
                                                type="textarea"
                                                :rows="2"
                                                :placeholder="`回复 @${reply.userName}`"
                                            ></el-input>
                                            <div class="reply-actions">
                                                <el-button size="small" @click="submitNestedReply(comment.commentId, reply)">
                                                    发送
                                                </el-button>
                                                <el-button size="small" type="text" @click="cancelNestedReply(comment.commentId, reply)">
                                                    取消
                                                </el-button>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <!-- 回复输入框 -->
                                <div v-if="replyInputs[comment.commentId]" class="reply-form">
                                    <el-input 
                                        v-model="replyContent[comment.commentId]"
                                        type="textarea"
                                        :rows="2"
                                        :placeholder="`回复 @${comment.userName}`"
                                    ></el-input>
                                    <div class="reply-actions">
                                        <el-button size="small" @click="submitCommentReply(comment)">
                                            发送
                                        </el-button>
                                        <el-button size="small" type="text" @click="cancelCommentReply(comment.commentId)">
                                            取消
                                        </el-button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </template>
                    <div v-else class="no-comments">
                        暂无评论，来抢沙发吧~
                    </div>
                </div>

                <!-- 分页器 -->
                <el-pagination
                    v-if="comments.length"
                    class="pagination"
                    background
                    layout="prev, pager, next"
                    :total="pagination.total"
                    :page-size="pagination.size"
                    :current-page="pagination.page"
                    @current-change="handlePageChange"
                />
            </div>
        </el-card>
        <div v-else class="loading-placeholder">
            <el-skeleton :rows="10" animated />
        </div>

        <!-- 悬浮点赞按钮 -->
        <div class="floating-actions">
            <div class="action-button like-button" 
                 :class="{ 'is-liked': isLiked }"
                 @click="toggleLike">
                <div class="button-content">
                    <el-icon size="24"><Star /></el-icon>
                    <span class="like-count">{{ likeCount }}</span>
                </div>
                <div class="tooltip">{{ isLiked ? '取消点赞' : '点赞' }}</div>
            </div>
            <div class="action-button comment-button" @click="scrollToComments">
                <div class="button-content">
                    <el-icon size="24"><ChatDotRound /></el-icon>
                    <span class="comment-count">{{ commentCount }}</span>
                </div>
                <div class="tooltip">评论</div>
            </div>
        </div>
    </div>
</template>

<script>
import { getNoteDetailService, addCommentService, getNoteCommentsService, getCommentRepliesService, getCommentCountService, deleteCommentService } from "@/api/note.js";
import { getUserInfoByIdService, getUserInfo } from "@/api/user.js";
import { ElMessage } from "element-plus";
import { dolikeNoteService, nolikeNoteService, doCollectService, unCollectService, noteCountService } from "@/api/collect.js";
import { 
    StarFilled, 
    ChatLineSquare,
    ChatDotRound,
    Star,
    Edit  // 可以用作表情图标
} from '@element-plus/icons-vue';

export default {
    data() {
        return {
            note: null, // 当前笔记情
            comments: [], // 评论列表
            newComment: { content: "" }, // 新评论内容
            replyInputs: {}, // 控制每个评论的回复输入框是否显示
            replyContent: {}, // 存储每个评论的回复内容
            isLiked: false,
            likeCount: 0,
            pagination: {
                page: 1,
                size: 10,
                total: 0
            },
            loading: {
                comments: false,
                replies: false,
                submit: false
            },
            activeReply: null, // 当前正在回复的评论
            commonEmojis: ['😊', '😂', '👍', '❤️', '🎉', '🤔', '👏', '🙏', '💪', '✨', 
                          '🌟', '🔥', '💯', '🎨', '💡', '💪', '🌈', '🎵', '🎸', '⚡️'],
            quickReplies: [
                '写得真好！支持一下~',
                '感谢分享，学习了！',
                '这个观点很有意思',
                '期待更多精彩内容',
                '赞同你的看法',
                '有深度，值得思考',
                '非常专业的见解',
                '说得对，收藏了',
                '分析得很到位',
                '继续加油！'
            ],
            commentCount: 0, // 添加评论总数
            currentUserId: null, // 当前登录用户ID
            isRepliesCollapsed: {}, // 控制评论的回复是否折叠
            showEmojiPicker: false, // 控制表情选择器的显示
        };
    },
    methods: {
        async fetchNoteCount() {
            const noteId = this.$route.params.id;
            if (!noteId) {
                console.warn("Note ID is invalid. Skipping fetchNoteCount.");
                return;
            }
            try {
                const response = await noteCountService({ noteId });
                console.log("帖子计数接口返回数据：", response);
                if (response.success) {
                    this.likeCount = response.data.likeTotal || 0;
                    console.log("点赞数：", this.likeCount);
                } else {
                    // 接口返回的成功标识为 false可以在这里设置默认值
                    console.warn("Failed to fetch note count, using defaults.");
                    this.likeCount = 0;
                }
            } catch (error) {
                // 捕获500错误，在控制台记录，但不弹错误提示，使用默认值兜底
                console.error("获取帖子计数信息失败，使用默认计数值:", error);
                this.likeCount = 0;
                // 可选：不使用 ElMessage 提示用户错误，将其默默处理
                // ElMessage.error("获取计数失败，已使用默认值");
            }
        },
        async fetchNoteDetail() {
            const noteId = this.$route.params.id;
            if (!noteId) {
                console.warn("Note ID is null or undefined. Skipping fetch.");
                return;
            }
            this.note = null;
            try {
                const response = await getNoteDetailService({ id: noteId });
                if (response.success) {
                    this.note = response.data;
                    this.isLiked = JSON.parse(localStorage.getItem(`note_${noteId}_liked`)) || false;
                    await this.fetchNoteCount(); // 获取计数信息
                    await this.fetchCommentCount(); // 添加这行
                    this.fetchComments();
                } else {
                    //ElMessage.error(response.message || "Failed to fetch note details.");
                    console.error("获取帖子计数信息失败，使用默认计数值:", error);
                    this.likeCount = 0;
                }
            } catch (error) {
                console.error("Error fetching note detail:", error);
                ElMessage.error("An error occurred while fetching note details.");
            }
        },
        async fetchComments() {
            const noteId = this.$route.params.id;
            this.loading.comments = true;
            try {
                const response = await getNoteCommentsService({ 
                    noteId, 
                    page: this.pagination.page, 
                    size: this.pagination.size 
                });
                
                if (response.success) {
                    console.log('Comments response:', response.data);
                    
                    if (response.data.records) {
                        this.comments = response.data.records;
                        this.comments.forEach(comment => {
                            console.log('Comment details:', {
                                commentId: comment.commentId,
                                userId: comment.userId,
                                userName: comment.userName,
                                currentUserId: this.currentUserId,
                                canDelete: this.isCurrentUser(comment.userId)
                            });
                        });
                        this.pagination.total = response.data.total;
                    } else {
                        this.comments = Array.isArray(response.data) ? response.data : [];
                        this.pagination.total = this.comments.length;
                    }
                    
                    console.log('处理后评论数据:', this.comments);
                    
                    if (this.comments.length > 0) {
                        await Promise.all(this.comments.map(comment => {
                            console.log('获取评论ID的回复:', comment.commentId);
                            return this.fetchReplies(comment.commentId);
                        }));
                    }
                } else {
                    ElMessage.warning(response.message || "获取评论失败");
                }
            } catch (error) {
                console.error("获取评论出错:", error);
                ElMessage.error("获取评论时发生错误");
            } finally {
                this.loading.comments = false;
            }
        },
        async fetchReplies(commentId) {
            if (!commentId) {
                console.warn('Invalid commentId:', commentId);
                return;
            }

            this.loading.replies = true;
            try {
                const response = await getCommentRepliesService({ commentId });
                console.log('Replies response for comment', commentId, ':', response);
                
                if (response.success) {
                    const commentIndex = this.comments.findIndex(
                        comment => comment.commentId === commentId
                    );
                    if (commentIndex !== -1) {
                        const replies = await Promise.all((response.data || []).map(async reply => {
                            let replyToUserName = this.comments[commentIndex].userName;
                            let replyToUserId = this.comments[commentIndex].userId;

                            // 如果回复的是其他回复，获取被回复用户的信息
                            if (reply.replyToUserId && reply.replyToUserId !== this.comments[commentIndex].userId) {
                                try {
                                    const userResponse = await getUserInfoByIdService({ id: reply.replyToUserId });
                                    if (userResponse.success) {
                                        replyToUserName = userResponse.data.nickName;
                                        replyToUserId = reply.replyToUserId;
                                    }
                                } catch (error) {
                                    console.error('获取被回复用户信息失败:', error);
                                }
                            }
                            
                            return {
                                ...reply,
                                replyToUserName,
                                replyToUserId
                            };
                        }));

                        this.comments[commentIndex] = {
                            ...this.comments[commentIndex],
                            replies: replies
                        };
                        
                        console.log('处理后的回复数据:', replies);
                    }
                }
            } catch (error) {
                console.error("获取回复出错:", error);
                ElMessage.error("获取回复失败");
            } finally {
                this.loading.replies = false;
            }
        },
        async submitComment() {
            if (!this.newComment.content.trim()) {
                ElMessage.warning("评论内容不能为空");
                return;
            }

            this.loading.submit = true;
            try {
                const response = await addCommentService({
                    noteId: this.$route.params.id,
                    content: this.newComment.content,
                    parentId: null // 一级评论无父评论
                });

                if (response.success) {
                    ElMessage.success("评论发布成功");
                    this.newComment.content = "";
                    this.pagination.page = 1;
                    await this.fetchCommentCount(); // 添加这行
                    await this.fetchComments();
                } else {
                    ElMessage.warning(response.message || "评论发布失败");
                }
            } catch (error) {
                console.error("发布评论出错:", error);
                ElMessage.error("发布评论时发生错误");
            } finally {
                this.loading.submit = false;
            }
        },
        showReplyInput(commentId) {
            this.replyInputs[commentId] = true;
        },
        async submitReply(parentCommentId) {
            const content = this.replyContent[parentCommentId]?.trim();
            if (!content) {
                ElMessage.warning("回复内容不能为空");
                return;
            }

            this.loading.submit = true;
            try {
                const response = await addCommentService({
                    noteId: this.$route.params.id,
                    content: content,
                    parentId: parentCommentId
                });

                if (response.success) {
                    ElMessage.success("回复发布成功");
                    this.replyContent[parentCommentId] = "";
                    this.replyInputs[parentCommentId] = false;
                    await this.fetchCommentCount(); // 添加这行
                    await this.fetchReplies(parentCommentId);
                } else {
                    ElMessage.warning(response.message || "回复发布失败");
                }
            } catch (error) {
                console.error("发布回复出错:", error);
                ElMessage.error("发布回复时发生错误");
            } finally {
                this.loading.submit = false;
            }
        },
        goToUserDetail(userId) {
            this.$router.push({ name: "UserDetail", params: { id: userId } });
        },
        async toggleLike() {
            try {
                if (this.isLiked) {
                    await nolikeNoteService(this.note.id)
                    this.likeCount--
                    ElMessage.success('已取消点赞')
                } else {
                    await dolikeNoteService(this.note.id)
                    this.likeCount++
                    ElMessage.success('点赞成功')
                }
                this.isLiked = !this.isLiked
            } catch (error) {
                console.error('Toggle like failed:', error)
                ElMessage.error('操作失败，请稍后重试')
            }
        },
        scrollToComments() {
            document.getElementById('comments')?.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            })
        },
        async handlePageChange(page) {
            this.pagination.page = page;
            await this.fetchComments();
        },
        // 换一级评论的回复输入框显示
        toggleCommentReply(comment) {
            this.replyInputs = {
                ...this.replyInputs,
                [comment.commentId]: !this.replyInputs[comment.commentId]
            };
        },

        // 取消一级评论回复
        cancelCommentReply(commentId) {
            this.replyInputs[commentId] = false;
            this.replyContent[commentId] = '';
        },

        // 提交一级评论的回复
        async submitCommentReply(comment) {
            if (!comment || !comment.commentId) {
                console.warn("Invalid comment object:", comment);
                return;
            }

            const replyContent = this.replyContent[comment.commentId];
            console.log('Reply content:', replyContent, 'for comment:', comment.commentId);

            if (!replyContent || !replyContent.trim()) {
                ElMessage.warning("回复内容不能为空");
                return;
            }

            this.loading.submit = true;
            try {
                const response = await addCommentService({
                    noteId: this.$route.params.id,
                    content: replyContent.trim(),
                    parentId: comment.commentId,
                    replyToUserId: comment.userId
                });

                if (response.success) {
                    ElMessage.success("回复发布成功");
                    this.replyContent[comment.commentId] = "";
                    this.replyInputs[comment.commentId] = false;
                    await this.fetchCommentCount();
                    await this.fetchReplies(comment.commentId);
                } else {
                    ElMessage.warning(response.message || "回复发布失败");
                }
            } catch (error) {
                console.error("发布回复出错:", error);
                ElMessage.error("发布回复时发生错误");
            } finally {
                this.loading.submit = false;
            }
        },

        // 为一级评论插入表情
        insertEmojiToComment(commentId, emoji) {
            const content = this.replyContent[commentId] || '';
            this.replyContent[commentId] = content + emoji;
        },

        // 修改原有的 toggleReplyInput 方法名，专门用于嵌套回复
        toggleNestedReply(commentId, reply) {
            const key = `${commentId}-${reply.id}`;
            this.replyInputs = {
                ...this.replyInputs,
                [key]: !this.replyInputs[key]
            };
            this.activeReply = reply;
            
            if (this.replyInputs[key]) {
                this.$nextTick(() => {
                    const input = this.$refs.replyInput;
                    if (input && input.focus) {
                        input.focus();
                    }
                });
            }
        },

        // 插入表情
        insertEmoji(emoji) {
            const textarea = document.querySelector('.comment-form .el-textarea__inner')
            if (textarea) {
                const start = textarea.selectionStart
                const end = textarea.selectionEnd
                const text = this.newComment.content
                this.newComment.content = text.substring(0, start) + emoji + text.substring(end)
                // 设置光标位置
                this.$nextTick(() => {
                    textarea.focus()
                    textarea.setSelectionRange(start + emoji.length, start + emoji.length)
                })
            }
        },

        // 插入快捷回复
        insertQuickReply(reply) {
            this.newComment.content = reply
        },

        // 快捷回复并直接提交
        async quickReplyAndSubmit(commentId, reply, content) {
            this.loading.submit = true;
            try {
                const response = await addCommentService({
                    noteId: this.$route.params.id,
                    content: content,
                    parentId: commentId,
                    replyToUserId: reply.userId
                });

                if (response.success) {
                    ElMessage.success("回复发布成功");
                    await this.fetchReplies(commentId);
                    await this.fetchCommentCount(); // 添加这行
                } else {
                    ElMessage.warning(response.message || "回复发布失败");
                }
            } catch (error) {
                console.error("发布回复出错:", error);
                ElMessage.error("发布回复发生错误");
            } finally {
                this.loading.submit = false;
            }
        },
        async fetchCommentCount() {
            const noteId = this.$route.params.id;
            try {
                const response = await getCommentCountService({ noteId });
                if (response.success) {
                    this.commentCount = response.data || 0;
                } else {
                    console.warn("Failed to fetch comment count");
                    this.commentCount = 0;
                }
            } catch (error) {
                console.error("Error fetching comment count:", error);
                this.commentCount = 0;
            }
        },
        // 提交嵌套回复
        async submitNestedReply(commentId, replyTo) {
            const key = `${commentId}-${replyTo.id}`;
            const content = this.replyContent[key]?.trim();
            
            if (!content) {
                ElMessage.warning("回复内容不能为空");
                return;
            }

            this.loading.submit = true;
            try {
                const response = await addCommentService({
                    noteId: this.$route.params.id,
                    content: content,
                    parentId: commentId,
                    replyToUserId: replyTo.userId,
                    replyToUserName: replyTo.userName // 添加被回复用户名
                });

                if (response.success) {
                    ElMessage.success("回复发布成功");
                    // 清空输入框和状态
                    this.replyContent[key] = "";
                    this.replyInputs[key] = false;
                    this.activeReply = null;
                    
                    // 重新获取评论数据
                    await this.fetchCommentCount();
                    await this.fetchReplies(commentId);
                } else {
                    ElMessage.warning(response.message || "回复发布失败");
                }
            } catch (error) {
                console.error("发布回复出错:", error);
                ElMessage.error("发布回复时发生错误");
            } finally {
                this.loading.submit = false;
            }
        },

        // 取消嵌套回复
        cancelNestedReply(commentId, reply) {
            if (!reply) return;
            
            const key = `${commentId}-${reply.id}`;
            this.replyInputs[key] = false;
            this.replyContent[key] = '';
            this.activeReply = null;
        },

        // 获取当前用户信息
        async getCurrentUser() {
            try {
                const response = await getUserInfo();
                console.log('Current user response:', response);

                if (response.success && response.data) {
                    this.currentUserId = response.data.id;
                    console.log('Set currentUserId to:', this.currentUserId);
                } else {
                    console.warn('Failed to get current user info:', response.message);
                    this.currentUserId = null;
                }
            } catch (error) {
                console.error('获取当前用户信息失败:', error);
                this.currentUserId = null;
            }
        },

        // 添加权限判断方法
        isCurrentUser(userId) {
            // 确保两个 ID 都转换为数字进行比较
            const currentId = Number(this.currentUserId);
            const targetId = Number(userId);
            
            const isMatch = currentId && targetId && currentId === targetId;
            console.log('Checking user permission:', {
                currentUserId: currentId,
                targetUserId: targetId,
                currentIdType: typeof currentId,
                targetIdType: typeof targetId,
                isMatch: isMatch
            });
            
            return isMatch;
        },

        // 修改删除方法，添加权限判断
        async deleteComment(commentId) {
            if (!this.currentUserId) {
                ElMessage.warning('请先登录');
                return;
            }

            console.log('尝试删除评论:', commentId);
            try {
                console.log('发送删除评论请求:', { commentId });
                const response = await deleteCommentService({ commentId });
                console.log('删除评论响应:', response);
                
                if (response.success) {
                    ElMessage.success('评论删除成功');
                    await this.fetchCommentCount();
                    await this.fetchComments();
                } else {
                    console.warn('删除评论失败:', response.message);
                    ElMessage.error(response.message || '删除失败');
                }
            } catch (error) {
                console.error('删除评论出错:', error);
                ElMessage.error('删除评论时发生错误');
            }
        },

        async deleteReply(commentId, replyId) {
            if (!this.currentUserId) {
                ElMessage.warning('请先登录');
                return;
            }

            console.log('试删除回复:', { commentId, replyId });
            try {
                console.log('发送删除回复请求:', { commentId: replyId });
                const response = await deleteCommentService({ commentId: replyId });
                console.log('删除回复响应:', response);
                
                if (response.success) {
                    ElMessage.success('回复删除成功');
                    await this.fetchCommentCount();
                    await this.fetchReplies(commentId);
                } else {
                    console.warn('删除回复失败:', response.message);
                    ElMessage.error(response.message || '删除失败');
                }
            } catch (error) {
                console.error('删除回复出错:', error);
                ElMessage.error('删除回复时发生错误');
            }
        },

        // 添加折叠功能
        toggleReplies(commentId) {
            this.isRepliesCollapsed = {
                ...this.isRepliesCollapsed,
                [commentId]: !this.isRepliesCollapsed[commentId]
            };
        }
    },
    async mounted() {
        await this.getCurrentUser();
        this.fetchNoteDetail();
    },
    watch: {
        "$route.params.id": {
            async handler(newId) {
                if (!newId) {
                    console.warn("Route param 'id' is invalid. Skipping fetch.");
                    this.note = null;
                    return;
                }
                this.note = null;
                await this.getCurrentUser(); // 重新获取用户信息
                await this.fetchNoteDetail();
            },
            immediate: true,
        },
    },
    components: {
        StarFilled,
        ChatLineSquare,
        ChatDotRound,
        Star,
        Edit
    }
};
</script>


<style scoped>
.page-container {
    padding: 24px;
    max-width: 1200px;
    margin: 0 auto;
    min-height: 100vh;
    background-color: #f5f7fa;
}

.content-card {
    background: white;
    border-radius: 12px;
}

/* 文章头部样式 */
.article-header {
    margin-bottom: 32px;
}

.article-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 24px;
    line-height: 1.3;
}

.article-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 24px;
    border-bottom: 1px solid #eee;
}

.author-info {
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
}

.author-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
}

.author-details {
    display: flex;
    flex-direction: column;
}

.author-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a1a;
}

.publish-time {
    font-size: 0.9rem;
    color: #666;
}

/* 图片轮播样式 */
.image-gallery {
    margin: 32px 0;
}

.carousel-image-wrapper {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f5f5f5;
    border-radius: 8px;
    overflow: hidden;
}

.carousel-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

/* 文章内容样式 */
.article-content {
    font-size: 1.1rem;
    line-height: 1.8;
    color: #333;
    margin: 32px 0;
}

/* 评论区样式 */
.comments-section {
    margin-top: 48px;
    padding-top: 32px;
    border-top: 1px solid #eee;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 24px;
}

.comment-form {
    margin-bottom: 32px;
}

.form-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 16px;
}

/* 评论列表样式 */
.comment-item {
    padding: 24px 0;
    border-bottom: 1px solid #eee;
}

.comment-user {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}

.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    cursor: pointer;
}

.user-info {
    display: flex;
    flex-direction: column;
}

.user-name {
    font-weight: 600;
    color: #1a1a1a;
}

.comment-time {
    font-size: 0.9rem;
    color: #666;
}

.comment-content {
    font-size: 1rem;
    line-height: 1.6;
    color: #333;
    margin: 12px 0;
}

/* 回复样式 */
.replies-list {
    margin-left: 52px;
    margin-top: 16px;
    padding: 16px;
    background: #f9f9f9;
    border-radius: 8px;
}

.reply-item {
    padding: 12px 0;
    border-bottom: 1px solid #eee;
}

.reply-item:last-child {
    border-bottom: none;
}

.reply-content {
    display: flex;
    gap: 8px;
    align-items: baseline;
    margin-bottom: 8px;
}

.reply-user {
    font-weight: 600;
    color: #1a1a1a;
    cursor: pointer;
}

.reply-user:hover {
    color: #409EFF;
}

.reply-arrow {
    color: #909399;
    font-size: 0.9em;
}

.reply-to-user {
    color: #409EFF;
    font-size: 0.9em;
}

.reply-text {
    color: #333;
}

.reply-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
}

.reply-time {
    font-size: 0.8rem;
    color: #909399;
}

.reply-actions {
    display: flex;
    gap: 12px;
}

.nested-reply-form {
    margin-top: 12px;
    padding: 12px;
    background: #fff;
    border-radius: 4px;
}

.nested-reply-form .reply-actions {
    margin-top: 8px;
    justify-content: flex-end;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .replies-list {
        margin-left: 24px;
        padding: 12px;
    }

    .reply-content {
        flex-wrap: wrap;
    }
}

/* 添加悬浮按钮样式 */
.floating-actions {
    position: fixed;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 16px;
    z-index: 100;
}

.action-button {
    position: relative;
    width: 48px;
    height: 48px;
    border-radius: 24px;
    background: white;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.button-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}

.like-count,
.comment-count {
    font-size: 12px;
    color: #666;
}

.action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.like-button {
    color: #666;
}

.like-button.is-liked {
    color: #409EFF;
    background: #ecf5ff;
}

.tooltip {
    position: absolute;
    right: calc(100% + 8px);
    top: 50%;
    transform: translateY(-50%);
    background: rgba(0, 0, 0, 0.75);
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    white-space: nowrap;
}

.action-button:hover .tooltip {
    opacity: 1;
    visibility: visible;
}

/* 添加动画效果 */
@keyframes likeAnimation {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

.like-button:active {
    animation: likeAnimation 0.3s ease;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .floating-actions {
        position: fixed;
        right: 16px;
        bottom: 24px;
        top: auto;
        transform: none;
        flex-direction: row;
    }

    .tooltip {
        display: none; /* 在移动端隐藏提示 */
    }
}

.comment-tools {
    display: flex;
    gap: 16px;
    padding: 12px 0;
    border-top: 1px solid #f0f0f0;
    margin-top: 8px;
}

.tool-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #666;
    transition: color 0.3s;
}

.tool-btn:hover {
    color: #409EFF;
}

.emoji-grid {
    display: none;
}

.emoji-item {
    display: none;
}

:deep(.emoji-dropdown) {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    width: 360px;
    padding: 8px;
}

:deep(.emoji-dropdown .el-dropdown-menu__item) {
    justify-content: center;
    padding: 8px;
    font-size: 24px;
    line-height: 1;
}

:deep(.emoji-dropdown .el-dropdown-menu__item:hover) {
    background-color: #f5f7fa;
    transform: scale(1.2);
    transition: all 0.2s ease;
}

:deep(.el-dropdown-menu__item) {
    line-height: 32px;
    padding: 0 16px;
    font-size: 14px;
}

:deep(.el-dropdown-menu__item:hover) {
    background-color: #f5f7fa;
    color: #409EFF;
}

/* 修改表情网格样式 */
:deep(.emoji-grid-menu) {
    width: 320px;
    padding: 8px;
    display: block !important;
}

:deep(.emoji-grid-container) {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 4px;
}

:deep(.emoji-grid-item) {
    height: 40px;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    line-height: 1;
    cursor: pointer;
    transition: all 0.2s ease;
}

:deep(.emoji-grid-item:hover) {
    background-color: #f5f7fa;
    transform: scale(1.2);
}

:deep(.emoji-grid-item.el-dropdown-menu__item) {
    min-width: unset;
    line-height: normal;
}

/* 移除之前的样式 */
:deep(.emoji-dropdown) {
    display: none;
}
</style>