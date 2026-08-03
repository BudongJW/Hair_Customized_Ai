package com.springboot.hair_customized_ai.storage;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public record PresignedUploadRequest(
    @NotBlank @Size(max = 255) String fileName,
    @NotBlank @Size(max = 100) String contentType,
    @NotNull UploadPurpose purpose
) {
}
