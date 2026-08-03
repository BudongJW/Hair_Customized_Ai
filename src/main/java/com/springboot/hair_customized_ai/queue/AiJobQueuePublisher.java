package com.springboot.hair_customized_ai.queue;

import java.util.UUID;

public interface AiJobQueuePublisher {

    void publishFaceProfilePreprocessing(UUID profileId);

    void publishHairFitting(UUID fittingJobId);
}
