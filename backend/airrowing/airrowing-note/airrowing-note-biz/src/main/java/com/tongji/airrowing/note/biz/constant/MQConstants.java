package com.tongji.airrowing.note.biz.constant;

public interface MQConstants {

    /**
     * Topic 主题：删除笔记本地缓存
     */
    String AIR_TOPIC_DELETE_NOTE_LOCAL_CACHE = "DeleteNoteLocalCacheTopic";

    /**
     * Topic: 点赞、取消点赞共用一个
     */
    String AIR_TOPIC_LIKE_OR_UNLIKE = "LikeUnlikeTopic";

    /**
     * Tag 标签：点赞
     */
    String AIR_TAG_LIKE = "Like";

    /**
     * Tag 标签：取消点赞
     */
    String AIR_TAG_UNLIKE = "Unlike";

    /**
     * Topic: 计数 - 笔记点赞数
     */
    String AIR_TOPIC_COUNT_NOTE_LIKE = "CountNoteLikeTopic";

    /**
     * Topic: 收藏、取消收藏共用一个
     */
    String AIR_TOPIC_COLLECT_OR_UN_COLLECT = "CollectUnCollectTopic";

    /**
     * Tag 标签：收藏
     */
    String AIR_TAG_COLLECT = "Collect";

    /**
     * Tag 标签：取消收藏
     */
    String AIR_TAG_UN_COLLECT = "UnCollect";

    /**
     * Topic: 计数 - 笔记收藏数
     */
    String AIR_TOPIC_COUNT_NOTE_COLLECT = "CountNoteCollectTopic";

    /**
     * Topic: 笔记操作（发布、删除）
     */
    String AIR_TOPIC_NOTE_OPERATE = "NoteOperateTopic";

    /**
     * Tag 标签：笔记发布
     */
    String AIR_TAG_NOTE_PUBLISH = "publishNote";

    /**
     * Tag 标签：笔记删除
     */
    String AIR_TAG_NOTE_DELETE = "deleteNote";
}



