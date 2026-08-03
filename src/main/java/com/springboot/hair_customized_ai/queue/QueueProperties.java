package com.springboot.hair_customized_ai.queue;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.queue")
public record QueueProperties(
    QueueProvider provider,
    String faceProfileQueue,
    String fittingJobQueue
) {
    public QueueProperties {
        if (provider == null) {
            provider = QueueProvider.LOG;
        }
        if (faceProfileQueue == null || faceProfileQueue.isBlank()) {
            faceProfileQueue = "ai.face-profile.preprocess";
        }
        if (fittingJobQueue == null || fittingJobQueue.isBlank()) {
            fittingJobQueue = "ai.fitting.generate";
        }
    }
}
