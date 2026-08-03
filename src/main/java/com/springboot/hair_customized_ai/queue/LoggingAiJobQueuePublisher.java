package com.springboot.hair_customized_ai.queue;

import java.util.UUID;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "app.queue", name = "provider", havingValue = "log", matchIfMissing = true)
public class LoggingAiJobQueuePublisher implements AiJobQueuePublisher {

    @Override
    public void publishFaceProfilePreprocessing(UUID profileId) {
        log.info("AI queue mock publish: {}", AiJobMessage.faceProfilePreprocessing(profileId));
    }

    @Override
    public void publishHairFitting(UUID fittingJobId) {
        log.info("AI queue mock publish: {}", AiJobMessage.hairFitting(fittingJobId));
    }
}
