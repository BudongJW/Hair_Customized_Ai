package com.springboot.hair_customized_ai.queue;

import java.time.OffsetDateTime;
import java.util.UUID;

public record AiJobMessage(
    AiJobType type,
    UUID aggregateId,
    OffsetDateTime requestedAt
) {
    public static AiJobMessage faceProfilePreprocessing(UUID profileId) {
        return new AiJobMessage(AiJobType.FACE_PROFILE_PREPROCESSING, profileId, OffsetDateTime.now());
    }

    public static AiJobMessage hairFitting(UUID fittingJobId) {
        return new AiJobMessage(AiJobType.HAIR_FITTING, fittingJobId, OffsetDateTime.now());
    }
}
