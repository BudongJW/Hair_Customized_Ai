package com.springboot.hair_customized_ai.storage;

public interface ObjectStorageService {

    PresignedUploadResponse createPresignedUpload(PresignedUploadRequest request);

    PresignedDownloadResponse createPresignedDownload(PresignedDownloadRequest request);
}
