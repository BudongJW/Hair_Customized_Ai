package com.springboot.hair_customized_ai.domain.profile;

public enum FaceProfileStatus {
    PENDING,
    PROCESSING,
    COMPLETED,
    FAILED;

    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED;
    }
}
