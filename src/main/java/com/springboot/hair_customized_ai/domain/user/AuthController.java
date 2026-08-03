package com.springboot.hair_customized_ai.domain.user;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final UserAccountService userAccountService;

    @PostMapping("/password/register")
    @ResponseStatus(HttpStatus.CREATED)
    AuthResponse register(@Valid @RequestBody PasswordRegisterRequest request) {
        return AuthResponse.from(userAccountService.registerWithPassword(
            request.email(),
            request.displayName(),
            request.password()
        ));
    }

    @PostMapping("/password/login")
    AuthResponse login(@Valid @RequestBody PasswordLoginRequest request) {
        return AuthResponse.from(userAccountService.loginWithPassword(request.email(), request.password()));
    }

    record PasswordRegisterRequest(
        @Email @NotBlank @Size(max = 255) String email,
        @NotBlank @Size(max = 80) String displayName,
        @NotBlank @Size(min = 6, max = 100) String password
    ) {
    }

    record PasswordLoginRequest(
        @Email @NotBlank @Size(max = 255) String email,
        @NotBlank @Size(min = 6, max = 100) String password
    ) {
    }

    record AuthResponse(
        UUID id,
        String email,
        String displayName,
        UserRole role,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {
        static AuthResponse from(UserAccount user) {
            return new AuthResponse(
                user.getId(),
                user.getEmail(),
                user.getDisplayName(),
                user.getRole(),
                user.getCreatedAt(),
                user.getUpdatedAt()
            );
        }
    }
}
