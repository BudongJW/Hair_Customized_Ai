package com.springboot.hair_customized_ai.storage;

import java.net.URI;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Configuration;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;

@Configuration
public class S3PresignerConfig {

    @Bean
    @ConditionalOnProperty(prefix = "app.storage", name = "mode", havingValue = "s3")
    S3Presigner s3Presigner(StorageProperties storageProperties) {
        var builder = S3Presigner.builder()
            .region(Region.of(storageProperties.region()))
            .serviceConfiguration(S3Configuration.builder()
                .pathStyleAccessEnabled(storageProperties.forcePathStyle())
                .build());

        if (StringUtils.hasText(storageProperties.endpoint())) {
            builder.endpointOverride(URI.create(storageProperties.endpoint()));
        }

        return builder.build();
    }
}
