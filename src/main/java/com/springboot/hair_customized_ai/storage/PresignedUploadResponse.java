package com.springboot.hair_customized_ai.storage;

import java.time.OffsetDateTime;
import java.util.Map;

public record PresignedUploadResponse(
    String uploadUrl,
    String objectKey,
    String publicUrl,
    OffsetDateTime expiresAt,
    Map<String, String> headers
) {
}
