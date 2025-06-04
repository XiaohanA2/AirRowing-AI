package com.tongji.airrowing.user.dto.req;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class FindUserByNicknameReqDTO {

    /**
     * 昵称
     */
    @NotBlank(message = "昵称不能为空")
    private String nickname;
}
