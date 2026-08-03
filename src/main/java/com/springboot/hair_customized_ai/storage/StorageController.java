package com.springboot.hair_customized_ai.storage;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Validated
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/uploads")
public class StorageController {

    private final ObjectStorageService objectStorageService;

    @PostMapping("/presigned-url")
    @ResponseStatus(HttpStatus.CREATED)
    PresignedUploadResponse createPresignedUpload(@Valid @RequestBody PresignedUploadRequest request) {
        return objectStorageService.createPresignedUpload(request);
    }

    @PostMapping("/presigned-download-url")
    PresignedDownloadResponse createPresignedDownload(@Valid @RequestBody PresignedDownloadRequest request) {
        return objectStorageService.createPresignedDownload(request);
    }
}
