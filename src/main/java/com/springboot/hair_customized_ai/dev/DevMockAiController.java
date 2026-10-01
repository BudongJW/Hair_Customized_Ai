package com.springboot.hair_customized_ai.dev;

import com.springboot.hair_customized_ai.domain.fitting.FittingJob;
import com.springboot.hair_customized_ai.domain.fitting.FittingJobService;
import com.springboot.hair_customized_ai.domain.fitting.FittingJobStatus;
import com.springboot.hair_customized_ai.domain.profile.FaceProfileStatus;
import com.springboot.hair_customized_ai.domain.profile.UserFaceProfile;
import com.springboot.hair_customized_ai.domain.profile.UserFaceProfileService;
import java.time.OffsetDateTime;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/dev/mock-ai")
@ConditionalOnProperty(prefix = "app.dev", name = "mock-ai-enabled", havingValue = "true", matchIfMissing = true)
public class DevMockAiController {

    private final UserFaceProfileService userFaceProfileService;
    private final FittingJobService fittingJobService;

    @PostMapping("/face-profiles/{profileId}/complete")
    MockFaceProfileResponse completeFaceProfile(@PathVariable UUID profileId) {
        UserFaceProfile profile = userFaceProfileService.updateAiResult(
            profileId,
            new UserFaceProfileService.FaceProfileAiResult(
                FaceProfileStatus.COMPLETED,
                "processed/face-profiles/%s/bald-canvas.webp".formatted(profileId),
                """
                    {"source":"mock","landmarkCount":468,"hairline":{"x":0.5,"y":0.22}}
                    """.trim(),
                0.8,
                -1.2,
                0.3,
                null
            )
        );
        return MockFaceProfileResponse.from(profile);
    }

    @PostMapping("/fitting-jobs/{jobId}/complete")
    MockFittingJobResponse completeFittingJob(@PathVariable UUID jobId) {
        FittingJob job = fittingJobService.updateAiResult(
            jobId,
            new FittingJobService.FittingJobAiResult(
                FittingJobStatus.COMPLETED,
                "results/fitting-jobs/%s/result.webp".formatted(jobId),
                "results/fitting-jobs/%s/hair-mask.png".formatted(jobId),
                "results/fitting-jobs/%s/hair-layer.png".formatted(jobId),
                null,
                null,
                null,
                null,
                null,
                null
            )
        );
        return MockFittingJobResponse.from(job);
    }

    record MockFaceProfileResponse(
        UUID id,
        UUID userId,
        FaceProfileStatus status,
        String baldCanvasObjectKey,
        OffsetDateTime completedAt,
        OffsetDateTime updatedAt
    ) {
        static MockFaceProfileResponse from(UserFaceProfile profile) {
            return new MockFaceProfileResponse(
                profile.getId(),
                profile.getUser().getId(),
                profile.getStatus(),
                profile.getBaldCanvasObjectKey(),
                profile.getCompletedAt(),
                profile.getUpdatedAt()
            );
        }
    }

    record MockFittingJobResponse(
        UUID id,
        UUID profileId,
        FittingJobStatus status,
        String resultImageObjectKey,
        OffsetDateTime completedAt,
        OffsetDateTime updatedAt
    ) {
        static MockFittingJobResponse from(FittingJob job) {
            return new MockFittingJobResponse(
                job.getId(),
                job.getProfile().getId(),
                job.getStatus(),
                job.getResultImageObjectKey(),
                job.getCompletedAt(),
                job.getUpdatedAt()
            );
        }
    }
}
