package com.springboot.hair_customized_ai.storage;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.storage")
public record StorageProperties(
    StorageMode mode,
    String bucket,
    String region,
    String endpoint,
    boolean forcePathStyle,
    Duration uploadUrlTtl,
    String publicBaseUrl,
    String mockUploadBaseUrl
) {
    public StorageProperties {
        if (mode == null) {
            mode = StorageMode.MOCK;
        }
        if (bucket == null || bucket.isBlank()) {
            bucket = "hair-customized-ai-dev";
        }
        if (region == null || region.isBlank()) {
            region = "ap-northeast-2";
        }
        if (uploadUrlTtl == null) {
            uploadUrlTtl = Duration.ofMinutes(10);
        }
        if (publicBaseUrl == null || publicBaseUrl.isBlank()) {
            publicBaseUrl = "http://localhost:8080/mock-storage";
        }
        if (mockUploadBaseUrl == null || mockUploadBaseUrl.isBlank()) {
            mockUploadBaseUrl = "http://localhost:8080/mock-upload";
        }
    }
}
