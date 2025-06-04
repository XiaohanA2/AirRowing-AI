package com.tongji.airrowing.user.relation.biz;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@MapperScan("com.tongji.airrowing.user.relation.biz.domain.mapper")
@EnableFeignClients(basePackages = "com.tongji.airrowing")
public class AirrowingUserRelationBizApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingUserRelationBizApplication.class, args);
    }

}
