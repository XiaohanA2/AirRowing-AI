package com.tongji.airrowing.auth;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;

@SpringBootApplication
@EnableFeignClients(basePackages = "com.tongji.airrowing")
public class AirrowingAuthApplication {

    public static void main(String[] args) {
        SpringApplication.run(AirrowingAuthApplication.class, args);
    }

}
