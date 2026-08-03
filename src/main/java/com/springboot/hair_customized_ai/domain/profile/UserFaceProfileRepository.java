package com.springboot.hair_customized_ai.domain.profile;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface UserFaceProfileRepository extends JpaRepository<UserFaceProfile, UUID> {

    @EntityGraph(attributePaths = "user")
    @Query("select profile from UserFaceProfile profile where profile.id = :profileId")
    Optional<UserFaceProfile> findWithUserById(@Param("profileId") UUID profileId);

    @EntityGraph(attributePaths = "user")
    List<UserFaceProfile> findByUser_IdOrderByCreatedAtDesc(UUID userId);

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query("""
        update UserFaceProfile profile
           set profile.status = :status,
               profile.baldCanvasObjectKey = :baldCanvasObjectKey,
               profile.landmarksJson = :landmarksJson,
               profile.yawDegrees = :yawDegrees,
               profile.pitchDegrees = :pitchDegrees,
               profile.rollDegrees = :rollDegrees,
               profile.failureReason = :failureReason,
               profile.completedAt = :completedAt,
               profile.updatedAt = :updatedAt
         where profile.id = :profileId
        """)
    int updateAiResult(
        @Param("profileId") UUID profileId,
        @Param("status") FaceProfileStatus status,
        @Param("baldCanvasObjectKey") String baldCanvasObjectKey,
        @Param("landmarksJson") String landmarksJson,
        @Param("yawDegrees") Double yawDegrees,
        @Param("pitchDegrees") Double pitchDegrees,
        @Param("rollDegrees") Double rollDegrees,
        @Param("failureReason") String failureReason,
        @Param("completedAt") OffsetDateTime completedAt,
        @Param("updatedAt") OffsetDateTime updatedAt
    );
}
