package com.springboot.hair_customized_ai.storage;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record PresignedDownloadRequest(
    @NotBlank @Size(max = 512) String objectKey
) {
}
