package com.springboot.hair_customized_ai.domain.profile;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/face-profiles")
public class UserFaceProfileController {

    private final UserFaceProfileService userFaceProfileService;

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    UserFaceProfileResponse create(@Valid @RequestBody CreateFaceProfileRequest request) {
        return UserFaceProfileResponse.from(
            userFaceProfileService.create(request.userId(), request.originalImageObjectKey())
        );
    }

    @GetMapping("/{profileId}")
    UserFaceProfileResponse get(@PathVariable UUID profileId) {
        return UserFaceProfileResponse.from(userFaceProfileService.get(profileId));
    }

    @GetMapping
    List<UserFaceProfileResponse> list(@RequestParam UUID userId) {
        return userFaceProfileService.listByUser(userId)
            .stream()
            .map(UserFaceProfileResponse::from)
            .toList();
    }

    @PatchMapping("/{profileId}/ai-result")
    UserFaceProfileResponse updateAiResult(
        @PathVariable UUID profileId,
        @Valid @RequestBody UpdateFaceProfileAiResultRequest request
    ) {
        return UserFaceProfileResponse.from(userFaceProfileService.updateAiResult(
            profileId,
            new UserFaceProfileService.FaceProfileAiResult(
                request.status(),
                request.baldCanvasObjectKey(),
                request.landmarksJson(),
                request.yawDegrees(),
                request.pitchDegrees(),
                request.rollDegrees(),
                request.failureReason()
            )
        ));
    }

    record CreateFaceProfileRequest(
        @NotNull UUID userId,
        @NotBlank @Size(max = 512) String originalImageObjectKey
    ) {
    }

    record UpdateFaceProfileAiResultRequest(
        @NotNull FaceProfileStatus status,
        @Size(max = 512) String baldCanvasObjectKey,
        String landmarksJson,
        Double yawDegrees,
        Double pitchDegrees,
        Double rollDegrees,
        @Size(max = 1000) String failureReason
    ) {
    }

    record UserFaceProfileResponse(
        UUID id,
        UUID userId,
        FaceProfileStatus status,
        String originalImageObjectKey,
        String baldCanvasObjectKey,
        String landmarksJson,
        Double yawDegrees,
        Double pitchDegrees,
        Double rollDegrees,
        String failureReason,
        OffsetDateTime completedAt,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {
        static UserFaceProfileResponse from(UserFaceProfile profile) {
            return new UserFaceProfileResponse(
                profile.getId(),
                profile.getUser().getId(),
                profile.getStatus(),
                profile.getOriginalImageObjectKey(),
                profile.getBaldCanvasObjectKey(),
                profile.getLandmarksJson(),
                profile.getYawDegrees(),
                profile.getPitchDegrees(),
                profile.getRollDegrees(),
                profile.getFailureReason(),
                profile.getCompletedAt(),
                profile.getCreatedAt(),
                profile.getUpdatedAt()
            );
        }
    }
}
