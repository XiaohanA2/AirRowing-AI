package com.tongji.framework.common.validator;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import java.lang.annotation.*;

@Target({ ElementType.METHOD, ElementType.FIELD, ElementType.ANNOTATION_TYPE, ElementType.PARAMETER })
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = EmailValidator.class)  // 指定自定义的验证器
public @interface Email {

    String message() default "邮箱格式不正确";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
