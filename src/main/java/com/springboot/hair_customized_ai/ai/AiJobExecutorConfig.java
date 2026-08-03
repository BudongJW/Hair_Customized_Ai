package com.springboot.hair_customized_ai.ai;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AiJobExecutorConfig {

    @Bean(destroyMethod = "shutdown")
    ExecutorService aiJobExecutor() {
        return Executors.newFixedThreadPool(2);
    }
}
