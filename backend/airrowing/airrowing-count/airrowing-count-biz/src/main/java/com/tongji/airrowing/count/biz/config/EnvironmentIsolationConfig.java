package com.tongji.airrowing.count.biz.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class EnvironmentIsolationConfig {

    @Value("${system.environment.name:}")  // 读取环境名称，默认空字符串
    private String environmentName;

    @Value("${system.environment.isolation:true}")  // 读取隔离开关，默认启用隔离
    private boolean isolationEnabled;

    // 获取隔离的 Topic 名称
    public String getIsolatedTopicName(String baseTopic) {
        if (isolationEnabled && environmentName != null && !environmentName.isEmpty()) {
            return baseTopic + "_" + environmentName;
        }
        return baseTopic;
    }

    // 获取隔离的 Consumer Group 名称
    public String getIsolatedConsumerGroup(String baseConsumerGroup) {
        if (isolationEnabled && environmentName != null && !environmentName.isEmpty()) {
            return baseConsumerGroup + "_" + environmentName;
        }
        return baseConsumerGroup;
    }
}
