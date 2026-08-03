package com.springboot.hair_customized_ai.domain.hair;

import com.springboot.hair_customized_ai.domain.user.UserAccount;
import com.springboot.hair_customized_ai.domain.user.UserAccountService;
import jakarta.persistence.EntityNotFoundException;
import java.util.List;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class HairDesignService {

    private final HairDesignRepository hairDesignRepository;
    private final UserAccountService userAccountService;

    @Transactional(readOnly = true)
    public HairDesign get(UUID hairDesignId) {
        return hairDesignRepository.findWithUserById(hairDesignId)
            .orElseThrow(() -> new EntityNotFoundException("Hair design not found: " + hairDesignId));
    }

    @Transactional(readOnly = true)
    public List<HairDesign> listByUser(UUID userId) {
        return hairDesignRepository.findByUser_IdOrderByCreatedAtDesc(userId);
    }

    @Transactional
    public HairDesign upsertAiResult(HairDesignAiResult result) {
        HairDesign design = hairDesignRepository.findBySourceFittingJobId(result.sourceFittingJobId())
            .orElseGet(() -> createDesign(result.userId(), result.sourceFittingJobId(), result.referenceImageObjectKey()));

        if (result.status() == HairDesignStatus.FAILED) {
            design.fail(result.failureReason());
            return design;
        }

        design.complete(
            result.hairMaskObjectKey(),
            result.hairLayerObjectKey(),
            result.previewImageObjectKey(),
            result.metadataJson()
        );
        return design;
    }

    private HairDesign createDesign(UUID userId, UUID sourceFittingJobId, String referenceImageObjectKey) {
        UserAccount user = userAccountService.get(userId);
        return hairDesignRepository.save(new HairDesign(user, sourceFittingJobId, referenceImageObjectKey));
    }

    public record HairDesignAiResult(
        UUID userId,
        UUID sourceFittingJobId,
        HairDesignStatus status,
        String referenceImageObjectKey,
        String hairMaskObjectKey,
        String hairLayerObjectKey,
        String previewImageObjectKey,
        String metadataJson,
        String failureReason
    ) {
    }
}
