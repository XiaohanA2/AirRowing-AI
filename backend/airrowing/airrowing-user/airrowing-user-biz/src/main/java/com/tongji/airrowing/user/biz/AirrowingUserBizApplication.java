package com.tongji.airrowing.user.biz;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@MapperScan("com.tongji.airrowing.user.biz.domain.mapper")
@EnableFeignClients(basePackages = "com.tongji.airrowing")
public class AirrowingUserBizApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingUserBizApplication.class, args);
    }
}
