package com.springboot.hair_customized_ai.domain.hair;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

public interface HairDesignRepository extends JpaRepository<HairDesign, UUID> {

    @EntityGraph(attributePaths = {"user"})
    Optional<HairDesign> findWithUserById(UUID hairDesignId);

    @EntityGraph(attributePaths = {"user"})
    Optional<HairDesign> findBySourceFittingJobId(UUID sourceFittingJobId);

    @EntityGraph(attributePaths = {"user"})
    List<HairDesign> findByUser_IdOrderByCreatedAtDesc(UUID userId);
}
