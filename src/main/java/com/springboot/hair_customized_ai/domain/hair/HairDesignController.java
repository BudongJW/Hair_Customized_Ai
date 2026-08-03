package com.springboot.hair_customized_ai.domain.hair;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/hair-designs")
public class HairDesignController {

    private final HairDesignService hairDesignService;

    @GetMapping("/{hairDesignId}")
    HairDesignResponse get(@PathVariable UUID hairDesignId) {
        return HairDesignResponse.from(hairDesignService.get(hairDesignId));
    }

    @GetMapping
    List<HairDesignResponse> list(@RequestParam UUID userId) {
        return hairDesignService.listByUser(userId)
            .stream()
            .map(HairDesignResponse::from)
            .toList();
    }

    @PatchMapping("/ai-result")
    HairDesignResponse upsertAiResult(@Valid @RequestBody UpdateHairDesignAiResultRequest request) {
        return HairDesignResponse.from(hairDesignService.upsertAiResult(
            new HairDesignService.HairDesignAiResult(
                request.userId(),
                request.sourceFittingJobId(),
                request.status(),
                request.referenceImageObjectKey(),
                request.hairMaskObjectKey(),
                request.hairLayerObjectKey(),
                request.previewImageObjectKey(),
                request.metadataJson(),
                request.failureReason()
            )
        ));
    }

    record UpdateHairDesignAiResultRequest(
        @NotNull UUID userId,
        @NotNull UUID sourceFittingJobId,
        @NotNull HairDesignStatus status,
        @NotBlank @Size(max = 512) String referenceImageObjectKey,
        @Size(max = 512) String hairMaskObjectKey,
        @Size(max = 512) String hairLayerObjectKey,
        @Size(max = 512) String previewImageObjectKey,
        String metadataJson,
        @Size(max = 1000) String failureReason
    ) {
    }

    record HairDesignResponse(
        UUID id,
        UUID userId,
        UUID sourceFittingJobId,
        HairDesignStatus status,
        String referenceImageObjectKey,
        String hairMaskObjectKey,
        String hairLayerObjectKey,
        String previewImageObjectKey,
        String metadataJson,
        String failureReason,
        OffsetDateTime completedAt,
        OffsetDateTime createdAt,
        OffsetDateTime updatedAt
    ) {
        static HairDesignResponse from(HairDesign design) {
            return new HairDesignResponse(
                design.getId(),
                design.getUser().getId(),
                design.getSourceFittingJobId(),
                design.getStatus(),
                design.getReferenceImageObjectKey(),
                design.getHairMaskObjectKey(),
                design.getHairLayerObjectKey(),
                design.getPreviewImageObjectKey(),
                design.getMetadataJson(),
                design.getFailureReason(),
                design.getCompletedAt(),
                design.getCreatedAt(),
                design.getUpdatedAt()
            );
        }
    }
}
