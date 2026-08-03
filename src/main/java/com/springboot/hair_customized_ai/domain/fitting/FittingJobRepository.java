package com.springboot.hair_customized_ai.domain.fitting;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface FittingJobRepository extends JpaRepository<FittingJob, UUID> {

    @EntityGraph(attributePaths = {"profile", "profile.user"})
    @Query("select job from FittingJob job where job.id = :jobId")
    java.util.Optional<FittingJob> findWithProfileById(@Param("jobId") UUID jobId);

    @EntityGraph(attributePaths = {"profile", "profile.user"})
    List<FittingJob> findByProfile_User_IdOrderByCreatedAtDesc(UUID userId);

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query("""
        update FittingJob job
           set job.status = :status,
               job.resultImageObjectKey = :resultImageObjectKey,
               job.hairMaskObjectKey = :hairMaskObjectKey,
               job.hairLayerObjectKey = :hairLayerObjectKey,
               job.hairDesignId = :hairDesignId,
               job.failureReason = :failureReason,
               job.completedAt = :completedAt,
               job.updatedAt = :updatedAt
         where job.id = :jobId
        """)
    int updateAiResult(
        @Param("jobId") UUID jobId,
        @Param("status") FittingJobStatus status,
        @Param("resultImageObjectKey") String resultImageObjectKey,
        @Param("hairMaskObjectKey") String hairMaskObjectKey,
        @Param("hairLayerObjectKey") String hairLayerObjectKey,
        @Param("hairDesignId") UUID hairDesignId,
        @Param("failureReason") String failureReason,
        @Param("completedAt") OffsetDateTime completedAt,
        @Param("updatedAt") OffsetDateTime updatedAt
    );
}
