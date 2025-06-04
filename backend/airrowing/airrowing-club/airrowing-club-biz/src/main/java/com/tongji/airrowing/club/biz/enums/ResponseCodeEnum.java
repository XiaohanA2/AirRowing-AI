package com.tongji.airrowing.club.biz.enums;

import com.tongji.framework.common.exception.BaseExceptionInterface;
import lombok.AllArgsConstructor;
import lombok.Getter;


@Getter
@AllArgsConstructor
public enum ResponseCodeEnum implements BaseExceptionInterface {

    // ----------- 通用异常状态码 -----------
    SYSTEM_ERROR("CLUB-10000", "出错啦，后台小哥正在努力修复中..."),
    PARAM_NOT_VALID("CLUB-10001", "参数错误"),
    ID_NOT_VALID("CLUB-10002", "身份检验失败"),

    // ----------- 业务异常状态码 -----------
    CLUB_NAME_EMPTY("CLUB-20000", "俱乐部名称不能为空"),
    CLUB_CREATION_FAILED("CLUB-20001", "创建俱乐部失败"),
    CLUB_NOT_FOUND("CLUB-20002", "没有这个俱乐部"),
    PERMISSION_DENIED("CLUB-20003", "权限不足"),
    ALREADY_A_HOST("CLUB-20004", "已经是主持人啦"),
    USER_NOT_FOUND("CLUB-20005", "用户不存在"),
    HOST_NOT_FOUND("CLUB-20006", "主持人不存在"),
    INVALID_TRANSFER("CLUB-20007", "仅可将身份转让给第二主持人"),
    INVALID_OPERATION("CLUB-20008", "无效操作"),
    DUPLICATE_JOIN_REQUEST("CLUB-20009", "请勿重复申请"),
    REQUEST_NOT_FOUND("CLUB-20010", "未找到申请或申请已处理"),
    MEMBER_NOT_FOUND("CLUB-20011", "你不是该俱乐部成员"),
    HOST_CANNOT_LEAVE("CLUB-20012", "主持人无法直接退出"),
    JOIN_REQUEST_DENIED("CLUB-20013", "已拒绝"),
    REQUEST_APPROVED("CLUB-20014", "已审核"),
    ACTIVITY_NOT_FOUND("CLUB-20015", "活动不存在"),
    ALREADY_APPLIED("CLUB-20016", "已报名"),
    SIGNUP_REJECTED("CLUB-20017", "报名被拒绝"),
    MEMBER_NOT_FOUND_IN_ACTIVITY("CLUB-20018", "该成员不在活动中"),
    MEMBER_NOT_FOUND_IN_CLUB("CLUB-20019", "该成员不在俱乐部中"),
    DATA_NOT_VALID("CLUB-20020", "请填写完整数据"),
    DATA_NOT_SAVED("CLUB-20021", "保存失败"),
    DATA_NOT_FIND("CLUB-20021", "未找到数据"),
    ;

    // 异常码
    private final String errorCode;
    // 错误信息
    private final String errorMessage;

}


