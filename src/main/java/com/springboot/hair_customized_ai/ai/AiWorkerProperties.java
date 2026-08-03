package com.springboot.hair_customized_ai.ai;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ai.worker")
public record AiWorkerProperties(
    String baseUrl,
    String backendBaseUrl
) {
    public AiWorkerProperties {
        if (baseUrl == null || baseUrl.isBlank()) {
            baseUrl = "http://localhost:8000";
        }
        if (backendBaseUrl == null || backendBaseUrl.isBlank()) {
            backendBaseUrl = "http://localhost:8080";
        }
    }
}
