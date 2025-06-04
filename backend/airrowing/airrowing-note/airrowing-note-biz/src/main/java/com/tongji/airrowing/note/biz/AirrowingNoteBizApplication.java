package com.tongji.airrowing.note.biz;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@MapperScan("com.tongji.airrowing.note.biz.domain.mapper")
@EnableFeignClients(basePackages = "com.tongji.airrowing")
public class AirrowingNoteBizApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingNoteBizApplication.class, args);
    }

}
