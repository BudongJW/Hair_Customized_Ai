package com.springboot.hair_customized_ai.domain.fitting;

public enum FittingJobStatus {
    PENDING,
    PROCESSING,
    COMPLETED,
    FAILED,
    REJECTED;

    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED || this == REJECTED;
    }
}
