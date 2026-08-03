package com.springboot.hair_customized_ai.domain.profile;

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
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Entity
@Table(
    name = "user_face_profiles",
    indexes = {
        @Index(name = "idx_face_profiles_user_id", columnList = "user_id"),
        @Index(name = "idx_face_profiles_status", columnList = "status")
    }
)
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class UserFaceProfile extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private UserAccount user;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 30)
    private FaceProfileStatus status = FaceProfileStatus.PENDING;

    @Column(nullable = false, length = 512)
    private String originalImageObjectKey;

    @Column(length = 512)
    private String baldCanvasObjectKey;

    @Column(columnDefinition = "text")
    private String landmarksJson;

    private Double yawDegrees;

    private Double pitchDegrees;

    private Double rollDegrees;

    @Column(length = 1000)
    private String failureReason;

    private OffsetDateTime completedAt;

    public UserFaceProfile(UserAccount user, String originalImageObjectKey) {
        this.user = user;
        this.originalImageObjectKey = originalImageObjectKey.trim();
    }
}
