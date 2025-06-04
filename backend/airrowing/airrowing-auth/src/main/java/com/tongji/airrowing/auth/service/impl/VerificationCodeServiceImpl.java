package com.tongji.airrowing.auth.service.impl;

import cn.hutool.core.util.RandomUtil;
import com.tongji.framework.common.exception.BizException;
import com.tongji.framework.common.response.Response;
import com.tongji.airrowing.auth.constant.RedisKeyConstants;
import com.tongji.airrowing.auth.enums.ResponseCodeEnum;
import com.tongji.airrowing.auth.model.vo.veriticationcode.SendVerificationCodeReqVO;
import com.tongji.airrowing.auth.service.VerificationCodeService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class VerificationCodeServiceImpl implements VerificationCodeService {

    @Resource
    private RedisTemplate<String, Object> redisTemplate;
    @Resource(name = "taskExecutor")
    private ThreadPoolTaskExecutor threadPoolTaskExecutor;
    @Autowired
    private JavaMailSender mailSender;  // 使用 JavaMailSender 发送邮件

    /**
     * 发送邮箱验证码
     *
     * @param sendVerificationCodeReqVO
     * @return
     */
    @Override
    public Response<?> send(SendVerificationCodeReqVO sendVerificationCodeReqVO) {
        // 获取邮箱
        String email = sendVerificationCodeReqVO.getEmail();
        log.info("==> 收到发送验证码请求，邮箱: {}", email);

        // 构建验证码 redis key
        String key = RedisKeyConstants.buildVerificationCodeKey(email);
        log.info("==> 生成的 Redis Key 为: {}", key);

        // 判断是否已发送验证码
        boolean isSent = redisTemplate.hasKey(key);
        log.info("==> 检查 Redis 中是否存在验证码, 结果: {}", isSent);
        if (isSent) {
            // 若之前发送的验证码未过期，则提示发送频繁
            throw new BizException(ResponseCodeEnum.VERIFICATION_CODE_SEND_FREQUENTLY);
        }

        // 生成 6 位随机数字验证码
        String verificationCode = RandomUtil.randomNumbers(6);

        log.info("==> 邮箱: {}, 已生成验证码：【{}】", email, verificationCode);

        // 调用邮件发送服务
        threadPoolTaskExecutor.submit(() -> {
            try {
                SimpleMailMessage message = new SimpleMailMessage();
                message.setTo(email);
                message.setSubject("您的验证码");
                message.setText("您的验证码是：" + verificationCode + "，有效期为3分钟，请勿泄露给他人。");
                message.setFrom("2900952121@qq.com");  // 替换为您的QQ邮箱
                log.info("==> 正在发送邮件至邮箱: {}", email);
                mailSender.send(message);
                log.info("==> 邮件已成功发送至: {}", email);
            } catch (Exception e) {
                log.error("邮件发送失败", e);
            }
        });

        // 存储验证码到 redis, 并设置过期时间为 3 分钟
        redisTemplate.opsForValue().set(key, verificationCode, 3, TimeUnit.MINUTES);
        log.info("==> 验证码已存入 Redis，Key: {}, 验证码: {}", key, verificationCode);


        return Response.success();
    }
}
