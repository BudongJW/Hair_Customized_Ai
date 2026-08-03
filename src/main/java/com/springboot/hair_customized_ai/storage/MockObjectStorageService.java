package com.springboot.hair_customized_ai.storage;

import java.time.OffsetDateTime;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "app.storage", name = "mode", havingValue = "mock", matchIfMissing = true)
public class MockObjectStorageService implements ObjectStorageService {

    private final StorageProperties storageProperties;
    private final ObjectKeyGenerator objectKeyGenerator;

    @Override
    public PresignedUploadResponse createPresignedUpload(PresignedUploadRequest request) {
        String objectKey = objectKeyGenerator.create(request.purpose(), request.fileName());
        OffsetDateTime expiresAt = OffsetDateTime.now().plus(storageProperties.uploadUrlTtl());
        return new PresignedUploadResponse(
            joinUrl(storageProperties.mockUploadBaseUrl(), objectKey),
            objectKey,
            joinUrl(storageProperties.publicBaseUrl(), objectKey),
            expiresAt,
            Map.of("Content-Type", request.contentType())
        );
    }

    @Override
    public PresignedDownloadResponse createPresignedDownload(PresignedDownloadRequest request) {
        return new PresignedDownloadResponse(
            joinUrl(storageProperties.publicBaseUrl(), request.objectKey()),
            request.objectKey(),
            OffsetDateTime.now().plus(storageProperties.uploadUrlTtl())
        );
    }

    private String joinUrl(String baseUrl, String objectKey) {
        return baseUrl.replaceAll("/+$", "") + "/" + objectKey;
    }
}
