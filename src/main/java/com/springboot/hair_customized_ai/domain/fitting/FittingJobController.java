package com.springboot.hair_customized_ai.domain.fitting;

import jakarta.validation.Valid;
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
@RequestMapping("/api/v1/fitting-jobs")
public class FittingJobController {

    private final FittingJobService fittingJobService;

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    FittingJobResponse create(@Valid @RequestBody CreateFittingJobRequest request) {
        return FittingJobResponse.from(
            fittingJobService.create(request.profileId(), request.referenceImageObjectKey(), request.hairDesignId())
        );
    }

    @GetMapping("/{jobId}")
    FittingJobResponse get(@PathVariable UUID jobId) {
        return FittingJobResponse.from(fittingJobService.get(jobId));
    }

    @GetMapping
    List<FittingJobResponse> list(@RequestParam UUID userId) {
        return fittingJobService.listByUser(userId)
            .stream()
            .map(FittingJobResponse::from)
            .toList();
    }

    @PatchMapping("/{jobId}/ai-result")
    FittingJobResponse updateAiResult(
        @PathVariable UUID jobId,
        @Valid @RequestBody UpdateFittingJobAiResultRequest request
    ) {
        return FittingJobResponse.from(fittingJobService.updateAiResult(
            jobId,
            new FittingJobService.FittingJobAiResult(
                request.status(),
                request.resultImageObjectKey(),
                request.hairMaskObjectKey(),
                request.hairLayerObjectKey(),
                request.hairDesignId(),
                request.failureReason()
            )
        ));
    }

    record CreateFittingJobRequest(
        @NotNull UUID profileId,
        @Size(max = 512) String referenceImageObjectKey,
        UUID hairDesignId
    ) {
    }

    record UpdateFittingJobAiResultRequest(
        @NotNull FittingJobStatus status,
        @Size(max = 512) String resultImageObjectKey,
        @Size(max = 512) String hairMaskObjectKey,
        @Size(max = 512) String hairLayerObjectKey,
        UUID hairDesignId,
        @Size(max = 1000) String failureReason
    ) {
    }

    record FittingJobResponse(
        UUID id,
        UUID profileId,
        UUID userId,
        FittingJobStatus status,
        String referenceImageObjectKey,
        UUID hairDesignId,
        String resultImageObjectKey,
        String hairMaskObjectKey,
        String hairLayerObjectKey,
        String failureReason,
        OffsetDateTime completedAt,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {
        static FittingJobResponse from(FittingJob job) {
            return new FittingJobResponse(
                job.getId(),
                job.getProfile().getId(),
                job.getProfile().getUser().getId(),
                job.getStatus(),
                job.getReferenceImageObjectKey(),
                job.getHairDesignId(),
                job.getResultImageObjectKey(),
                job.getHairMaskObjectKey(),
                job.getHairLayerObjectKey(),
                job.getFailureReason(),
                job.getCompletedAt(),
                job.getCreatedAt(),
                job.getUpdatedAt()
            );
        }
    }
}
