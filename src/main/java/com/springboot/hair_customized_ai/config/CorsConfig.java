package com.springboot.hair_customized_ai.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
            .allowedOriginPatterns("*")
            .allowedMethods("GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS")
            .allowedHeaders("*")
            .maxAge(3600);

        registry.addMapping("/mock-upload/**")
            .allowedOriginPatterns("*")
            .allowedMethods("PUT", "OPTIONS")
            .allowedHeaders("*")
            .maxAge(3600);
    }
}
