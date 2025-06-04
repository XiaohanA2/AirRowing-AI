# 使用轻量级的 OpenJDK 镜像
FROM openjdk:17-jdk-slim

# 将构建好的 JAR 文件复制到镜像中
COPY target/airrowing-auth-0.0.1-SNAPSHOT.jar /app/airrowing-auth.jar

# 暴露应用运行端口
EXPOSE 8080

# 启动服务
ENTRYPOINT ["java", "-jar", "/app/airrowing-auth.jar"]
