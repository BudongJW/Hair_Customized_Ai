package com.springboot.hair_customized_ai.domain.hair;

import com.springboot.hair_customized_ai.domain.BaseEntity;
import com.springboot.hair_customized_ai.domain.user.UserAccount;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.Index;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.OffsetDateTime;
import java.util.UUID;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Entity
@Table(
    name = "hair_designs",
    indexes = {
        @Index(name = "idx_hair_designs_user_id", columnList = "user_id"),
        @Index(name = "idx_hair_designs_source_fitting_job_id", columnList = "source_fitting_job_id"),
        @Index(name = "idx_hair_designs_status", columnList = "status")
    }
)
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class HairDesign extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private UserAccount user;

    @Column(name = "source_fitting_job_id", unique = true)
    private UUID sourceFittingJobId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 30)
    private HairDesignStatus status;

    @Column(nullable = false, length = 512)
    private String referenceImageObjectKey;

    @Column(length = 512)
    private String hairMaskObjectKey;

    @Column(length = 512)
    private String hairLayerObjectKey;

    @Column(length = 512)
    private String previewImageObjectKey;

    @Column(columnDefinition = "text")
    private String metadataJson;

    @Column(length = 1000)
    private String failureReason;

    private OffsetDateTime completedAt;

    public HairDesign(UserAccount user, UUID sourceFittingJobId, String referenceImageObjectKey) {
        this.user = user;
        this.sourceFittingJobId = sourceFittingJobId;
        this.referenceImageObjectKey = referenceImageObjectKey.trim();
        this.status = HairDesignStatus.COMPLETED;
    }

    public void complete(
        String hairMaskObjectKey,
        String hairLayerObjectKey,
        String previewImageObjectKey,
        String metadataJson
    ) {
        this.status = HairDesignStatus.COMPLETED;
        this.hairMaskObjectKey = trimToNull(hairMaskObjectKey);
        this.hairLayerObjectKey = trimToNull(hairLayerObjectKey);
        this.previewImageObjectKey = trimToNull(previewImageObjectKey);
        this.metadataJson = metadataJson;
        this.failureReason = null;
        this.completedAt = OffsetDateTime.now();
    }

    public void fail(String failureReason) {
        this.status = HairDesignStatus.FAILED;
        this.failureReason = trimToNull(failureReason);
        this.completedAt = OffsetDateTime.now();
    }

    private String trimToNull(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return value.trim();
    }
}
