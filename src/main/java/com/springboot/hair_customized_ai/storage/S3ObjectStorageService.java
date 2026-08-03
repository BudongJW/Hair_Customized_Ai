package com.springboot.hair_customized_ai.storage;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.PresignedGetObjectRequest;
import software.amazon.awssdk.services.s3.presigner.model.PresignedPutObjectRequest;

@Service
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "app.storage", name = "mode", havingValue = "s3")
public class S3ObjectStorageService implements ObjectStorageService {

    private final S3Presigner s3Presigner;
    private final StorageProperties storageProperties;
    private final ObjectKeyGenerator objectKeyGenerator;

    @Override
    public PresignedUploadResponse createPresignedUpload(PresignedUploadRequest request) {
        String objectKey = objectKeyGenerator.create(request.purpose(), request.fileName());
        PutObjectRequest putObjectRequest = PutObjectRequest.builder()
            .bucket(storageProperties.bucket())
            .key(objectKey)
            .contentType(request.contentType())
            .build();

        PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
            .signatureDuration(storageProperties.uploadUrlTtl())
            .putObjectRequest(putObjectRequest)
            .build();

        PresignedPutObjectRequest presignedRequest = s3Presigner.presignPutObject(presignRequest);
        return new PresignedUploadResponse(
            presignedRequest.url().toExternalForm(),
            objectKey,
            publicUrl(objectKey),
            OffsetDateTime.ofInstant(presignedRequest.expiration(), ZoneOffset.UTC),
            flattenHeaders(presignedRequest.signedHeaders())
        );
    }

    @Override
    public PresignedDownloadResponse createPresignedDownload(PresignedDownloadRequest request) {
        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
            .bucket(storageProperties.bucket())
            .key(request.objectKey())
            .build();

        GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
            .signatureDuration(storageProperties.uploadUrlTtl())
            .getObjectRequest(getObjectRequest)
            .build();

        PresignedGetObjectRequest presignedRequest = s3Presigner.presignGetObject(presignRequest);
        return new PresignedDownloadResponse(
            presignedRequest.url().toExternalForm(),
            request.objectKey(),
            OffsetDateTime.ofInstant(presignedRequest.expiration(), ZoneOffset.UTC)
        );
    }

    private String publicUrl(String objectKey) {
        return storageProperties.publicBaseUrl().replaceAll("/+$", "") + "/" + objectKey;
    }

    private Map<String, String> flattenHeaders(Map<String, List<String>> headers) {
        Map<String, String> flattened = new LinkedHashMap<>();
        headers.forEach((name, values) -> flattened.put(name, String.join(",", values)));
        return flattened;
    }
}
