package com.tongji.airrowing.oss.biz.service;

import com.tongji.framework.common.response.Response;
import org.springframework.web.multipart.MultipartFile;

public interface FileService {

    /**
     * 上传文件
     *
     * @param file
     * @return
     */
    Response<?> uploadFile(MultipartFile file);
}

