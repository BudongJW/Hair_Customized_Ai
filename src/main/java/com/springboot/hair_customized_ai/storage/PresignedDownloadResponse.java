package com.springboot.hair_customized_ai.storage;

import java.time.OffsetDateTime;

public record PresignedDownloadResponse(
    String downloadUrl,
    String objectKey,
    OffsetDateTime expiresAt
) {
}
