package com.springboot.hair_customized_ai.domain.fitting;

import com.springboot.hair_customized_ai.common.tx.AfterCommitExecutor;
import com.springboot.hair_customized_ai.domain.hair.HairDesign;
import com.springboot.hair_customized_ai.domain.hair.HairDesignService;
import com.springboot.hair_customized_ai.domain.profile.FaceProfileStatus;
import com.springboot.hair_customized_ai.domain.profile.UserFaceProfile;
import com.springboot.hair_customized_ai.domain.profile.UserFaceProfileService;
import com.springboot.hair_customized_ai.queue.AiJobQueuePublisher;
import jakarta.persistence.EntityNotFoundException;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class FittingJobService {

    private final FittingJobRepository fittingJobRepository;
    private final UserFaceProfileService userFaceProfileService;
    private final HairDesignService hairDesignService;
    private final AiJobQueuePublisher aiJobQueuePublisher;
    private final AfterCommitExecutor afterCommitExecutor;

    @Transactional
    public FittingJob create(UUID profileId, String referenceImageObjectKey, UUID hairDesignId) {
        UserFaceProfile profile = userFaceProfileService.get(profileId);
        if (profile.getStatus() != FaceProfileStatus.COMPLETED) {
            throw new IllegalStateException("Face profile must be completed before fitting");
        }

        String resolvedReferenceImageObjectKey = resolveReferenceImageObjectKey(profile, referenceImageObjectKey, hairDesignId);
        FittingJob job = fittingJobRepository.save(new FittingJob(profile, resolvedReferenceImageObjectKey, hairDesignId));
        afterCommitExecutor.run(() -> aiJobQueuePublisher.publishHairFitting(job.getId()));
        return job;
    }

    @Transactional(readOnly = true)
    public FittingJob get(UUID jobId) {
        return fittingJobRepository.findWithProfileById(jobId)
            .orElseThrow(() -> new EntityNotFoundException("Fitting job not found: " + jobId));
    }

    @Transactional(readOnly = true)
    public List<FittingJob> listByUser(UUID userId) {
        return fittingJobRepository.findByProfile_User_IdOrderByCreatedAtDesc(userId);
    }

    @Transactional
    public FittingJob updateAiResult(UUID jobId, FittingJobAiResult result) {
        OffsetDateTime now = OffsetDateTime.now();
        OffsetDateTime completedAt = result.status().isTerminal() ? now : null;
        int updatedRows = fittingJobRepository.updateAiResult(
            jobId,
            result.status(),
            result.resultImageObjectKey(),
            result.hairMaskObjectKey(),
            result.hairLayerObjectKey(),
            result.targetHairMaskObjectKey(),
            result.inpaintingMaskObjectKey(),
            result.faceProtectionMaskObjectKey(),
            result.pipelineManifestObjectKey(),
            result.hairDesignId(),
            result.failureReason(),
            completedAt,
            now
        );
        if (updatedRows == 0) {
            throw new EntityNotFoundException("Fitting job not found: " + jobId);
        }
        return get(jobId);
    }

    private String resolveReferenceImageObjectKey(
        UserFaceProfile profile,
        String referenceImageObjectKey,
        UUID hairDesignId
    ) {
        if (hairDesignId == null) {
            if (referenceImageObjectKey == null || referenceImageObjectKey.isBlank()) {
                throw new IllegalArgumentException("referenceImageObjectKey is required when hairDesignId is not provided");
            }
            return referenceImageObjectKey;
        }

        HairDesign design = hairDesignService.get(hairDesignId);
        if (!design.getUser().getId().equals(profile.getUser().getId())) {
            throw new IllegalArgumentException("Hair design belongs to a different user");
        }
        return design.getReferenceImageObjectKey();
    }

    public record FittingJobAiResult(
        FittingJobStatus status,
        String resultImageObjectKey,
        String hairMaskObjectKey,
        String hairLayerObjectKey,
        String targetHairMaskObjectKey,
        String inpaintingMaskObjectKey,
        String faceProtectionMaskObjectKey,
        String pipelineManifestObjectKey,
        UUID hairDesignId,
        String failureReason
    ) {
    }
}
