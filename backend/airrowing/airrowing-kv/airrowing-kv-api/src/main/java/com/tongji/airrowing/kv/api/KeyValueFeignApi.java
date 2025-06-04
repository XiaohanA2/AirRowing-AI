package com.tongji.airrowing.kv.api;

import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.kv.constant.ApiConstants;
import com.tongji.airrowing.kv.dto.req.AddNoteContentReqDTO;
import com.tongji.airrowing.kv.dto.req.DeleteNoteContentReqDTO;
import com.tongji.airrowing.kv.dto.req.FindNoteContentReqDTO;
import com.tongji.airrowing.kv.dto.rsp.FindNoteContentRspDTO;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

@FeignClient(name = ApiConstants.SERVICE_NAME)
public interface KeyValueFeignApi {

    String PREFIX = "/kv";

    @PostMapping(value = PREFIX + "/note/content/add")
    Response<?> addNoteContent(@RequestBody AddNoteContentReqDTO addNoteContentReqDTO);

    @PostMapping(value = PREFIX + "/note/content/find")
    Response<FindNoteContentRspDTO> findNoteContent(@RequestBody FindNoteContentReqDTO findNoteContentReqDTO);

    @PostMapping(value = PREFIX + "/note/content/delete")
    Response<?> deleteNoteContent(@RequestBody DeleteNoteContentReqDTO deleteNoteContentReqDTO);
}


