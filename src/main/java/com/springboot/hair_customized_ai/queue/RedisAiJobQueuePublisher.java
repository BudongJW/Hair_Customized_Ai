package com.springboot.hair_customized_ai.queue;

import java.util.UUID;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "app.queue", name = "provider", havingValue = "redis")
public class RedisAiJobQueuePublisher implements AiJobQueuePublisher {

    private final StringRedisTemplate redisTemplate;
    private final QueueProperties queueProperties;

    @Override
    public void publishFaceProfilePreprocessing(UUID profileId) {
        publish(queueProperties.faceProfileQueue(), AiJobMessage.faceProfilePreprocessing(profileId));
    }

    @Override
    public void publishHairFitting(UUID fittingJobId) {
        publish(queueProperties.fittingJobQueue(), AiJobMessage.hairFitting(fittingJobId));
    }

    private void publish(String queueName, AiJobMessage message) {
        redisTemplate.opsForList().rightPush(queueName, serialize(message));
    }

    private String serialize(AiJobMessage message) {
        return """
            {"type":"%s","aggregateId":"%s","requestedAt":"%s"}
            """.formatted(message.type(), message.aggregateId(), message.requestedAt()).trim();
    }
}
