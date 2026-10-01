package com.springboot.hair_customized_ai.domain.fitting;

public enum FittingJobStatus {
    PENDING,
    PROCESSING,
    PREPARED,
    COMPLETED,
    FAILED,
    REJECTED;

    public boolean isTerminal() {
        return this == PREPARED || this == COMPLETED || this == FAILED || this == REJECTED;
    }
}
