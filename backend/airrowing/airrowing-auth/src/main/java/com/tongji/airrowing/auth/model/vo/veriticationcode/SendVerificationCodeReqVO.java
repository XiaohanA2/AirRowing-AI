package com.tongji.airrowing.auth.model.vo.veriticationcode;
import com.tongji.framework.common.validator.Email;
import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class SendVerificationCodeReqVO {

    @NotBlank(message = "邮箱不能为空")
    @Email
    private String email;

}

