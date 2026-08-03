package com.springboot.hair_customized_ai.domain.fitting;

import com.springboot.hair_customized_ai.domain.BaseEntity;
import com.springboot.hair_customized_ai.domain.profile.UserFaceProfile;
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
    name = "fitting_jobs",
    indexes = {
        @Index(name = "idx_fitting_jobs_profile_id", columnList = "profile_id"),
        @Index(name = "idx_fitting_jobs_status", columnList = "status")
    }
)
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class FittingJob extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "profile_id", nullable = false)
    private UserFaceProfile profile;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 30)
    private FittingJobStatus status = FittingJobStatus.PENDING;

    @Column(nullable = false, length = 512)
    private String referenceImageObjectKey;

    private UUID hairDesignId;

    @Column(length = 512)
    private String resultImageObjectKey;

    @Column(length = 512)
    private String hairMaskObjectKey;

    @Column(length = 512)
    private String hairLayerObjectKey;

    @Column(length = 1000)
    private String failureReason;

    private OffsetDateTime completedAt;

    public FittingJob(UserFaceProfile profile, String referenceImageObjectKey) {
        this(profile, referenceImageObjectKey, null);
    }

    public FittingJob(UserFaceProfile profile, String referenceImageObjectKey, UUID hairDesignId) {
        this.profile = profile;
        this.referenceImageObjectKey = referenceImageObjectKey.trim();
        this.hairDesignId = hairDesignId;
    }
}
