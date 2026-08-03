package com.springboot.hair_customized_ai.domain.profile;

import com.springboot.hair_customized_ai.common.tx.AfterCommitExecutor;
import com.springboot.hair_customized_ai.domain.user.UserAccount;
import com.springboot.hair_customized_ai.domain.user.UserAccountService;
import com.springboot.hair_customized_ai.queue.AiJobQueuePublisher;
import jakarta.persistence.EntityNotFoundException;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class UserFaceProfileService {

    private final UserFaceProfileRepository userFaceProfileRepository;
    private final UserAccountService userAccountService;
    private final AiJobQueuePublisher aiJobQueuePublisher;
    private final AfterCommitExecutor afterCommitExecutor;

    @Transactional
    public UserFaceProfile create(UUID userId, String originalImageObjectKey) {
        UserAccount user = userAccountService.get(userId);
        UserFaceProfile profile = userFaceProfileRepository.save(
            new UserFaceProfile(user, originalImageObjectKey)
        );

        afterCommitExecutor.run(() -> aiJobQueuePublisher.publishFaceProfilePreprocessing(profile.getId()));
        return profile;
    }

    @Transactional(readOnly = true)
    public UserFaceProfile get(UUID profileId) {
        return userFaceProfileRepository.findWithUserById(profileId)
            .orElseThrow(() -> new EntityNotFoundException("Face profile not found: " + profileId));
    }

    @Transactional(readOnly = true)
    public List<UserFaceProfile> listByUser(UUID userId) {
        userAccountService.get(userId);
        return userFaceProfileRepository.findByUser_IdOrderByCreatedAtDesc(userId);
    }

    @Transactional
    public UserFaceProfile updateAiResult(UUID profileId, FaceProfileAiResult result) {
        OffsetDateTime now = OffsetDateTime.now();
        OffsetDateTime completedAt = result.status().isTerminal() ? now : null;
        int updatedRows = userFaceProfileRepository.updateAiResult(
            profileId,
            result.status(),
            result.baldCanvasObjectKey(),
            result.landmarksJson(),
            result.yawDegrees(),
            result.pitchDegrees(),
            result.rollDegrees(),
            result.failureReason(),
            completedAt,
            now
        );
        if (updatedRows == 0) {
            throw new EntityNotFoundException("Face profile not found: " + profileId);
        }
        return get(profileId);
    }

    public record FaceProfileAiResult(
        FaceProfileStatus status,
        String baldCanvasObjectKey,
        String landmarksJson,
        Double yawDegrees,
        Double pitchDegrees,
        Double rollDegrees,
        String failureReason
    ) {
    }
}
