package com.springboot.hair_customized_ai.common.web;

import java.time.OffsetDateTime;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/health")
public class HealthController {

    @GetMapping
    HealthResponse health() {
        return new HealthResponse("ok", OffsetDateTime.now());
    }

    record HealthResponse(String status, OffsetDateTime checkedAt) {
    }
}
