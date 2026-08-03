package com.springboot.hair_customized_ai.queue;

import com.springboot.hair_customized_ai.ai.AiWorkerProperties;
import java.util.concurrent.ExecutorService;
import java.util.UUID;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

@Slf4j
@Component
@ConditionalOnProperty(prefix = "app.queue", name = "provider", havingValue = "http")
public class HttpAiJobQueuePublisher implements AiJobQueuePublisher {

    private final AiWorkerProperties aiWorkerProperties;
    private final ExecutorService aiJobExecutor;
    private final RestClient restClient;

    public HttpAiJobQueuePublisher(
        AiWorkerProperties aiWorkerProperties,
        @Qualifier("aiJobExecutor") ExecutorService aiJobExecutor
    ) {
        this.aiWorkerProperties = aiWorkerProperties;
        this.aiJobExecutor = aiJobExecutor;
        this.restClient = RestClient.builder()
            .baseUrl(aiWorkerProperties.baseUrl())
            .build();
    }

    @Override
    public void publishFaceProfilePreprocessing(UUID profileId) {
        aiJobExecutor.submit(() -> requestFaceProfilePreprocessing(profileId));
    }

    @Override
    public void publishHairFitting(UUID fittingJobId) {
        aiJobExecutor.submit(() -> requestHairFitting(fittingJobId));
    }

    private void requestFaceProfilePreprocessing(UUID profileId) {
        try {
            restClient.post()
                .uri("/api/v1/jobs/face-profile-preprocessing")
                .body(new FaceProfileJobRequest(profileId, aiWorkerProperties.backendBaseUrl()))
                .retrieve()
                .toBodilessEntity();
        } catch (RestClientException ex) {
            log.error("Failed to request face profile AI job: profileId={}", profileId, ex);
        }
    }

    private void requestHairFitting(UUID fittingJobId) {
        try {
            restClient.post()
                .uri("/api/v1/jobs/hair-fitting")
                .body(new HairFittingJobRequest(fittingJobId, aiWorkerProperties.backendBaseUrl()))
                .retrieve()
                .toBodilessEntity();
        } catch (RestClientException ex) {
            log.error("Failed to request hair fitting AI job: fittingJobId={}", fittingJobId, ex);
        }
    }

    record FaceProfileJobRequest(UUID profileId, String backendBaseUrl) {
    }

    record HairFittingJobRequest(UUID fittingJobId, String backendBaseUrl) {
    }
}
