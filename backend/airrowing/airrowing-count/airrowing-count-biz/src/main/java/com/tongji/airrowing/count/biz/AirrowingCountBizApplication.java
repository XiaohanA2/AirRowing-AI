package com.tongji.airrowing.count.biz;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.tongji.airrowing.count.biz.domain.mapper")
public class AirrowingCountBizApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingCountBizApplication.class, args);
    }

}
