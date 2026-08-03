package com.springboot.hair_customized_ai.storage;

import java.time.LocalDate;
import java.util.Locale;
import java.util.UUID;
import org.springframework.stereotype.Component;

@Component
public class ObjectKeyGenerator {

    public String create(UploadPurpose purpose, String fileName) {
        LocalDate today = LocalDate.now();
        return "uploads/%s/%s/%s-%s".formatted(
            purpose.name().toLowerCase(Locale.ROOT).replace('_', '-'),
            today,
            UUID.randomUUID(),
            sanitize(fileName)
        );
    }

    private String sanitize(String fileName) {
        String sanitized = fileName.trim()
            .replace('\\', '/')
            .replaceAll(".*/", "")
            .replaceAll("[^A-Za-z0-9._-]", "-");
        return sanitized.isBlank() ? "upload.bin" : sanitized;
    }
}
