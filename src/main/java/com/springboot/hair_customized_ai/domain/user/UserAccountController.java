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
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/users")
public class UserAccountController {

    private final UserAccountService userAccountService;

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    UserAccountResponse create(@Valid @RequestBody CreateUserAccountRequest request) {
        return UserAccountResponse.from(userAccountService.create(request.email(), request.displayName()));
    }

    @GetMapping("/{userId}")
    UserAccountResponse get(@PathVariable UUID userId) {
        return UserAccountResponse.from(userAccountService.get(userId));
    }

    record CreateUserAccountRequest(
        @Email @NotBlank @Size(max = 255) String email,
        @NotBlank @Size(max = 80) String displayName
    ) {
    }

    record UserAccountResponse(
        UUID id,
        String email,
        String displayName,
        UserRole role,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {
        static UserAccountResponse from(UserAccount user) {
            return new UserAccountResponse(
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
