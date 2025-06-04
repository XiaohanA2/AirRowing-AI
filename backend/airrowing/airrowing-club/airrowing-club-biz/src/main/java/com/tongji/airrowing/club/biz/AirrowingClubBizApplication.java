package com.tongji.airrowing.club.biz;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@MapperScan("com.tongji.airrowing.club.biz.domain.mapper")
@EnableFeignClients(basePackages = "com.tongji.airrowing")
public class AirrowingClubBizApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingClubBizApplication.class, args);
    }

}
