package com.springboot.hair_customized_ai.storage;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@ConditionalOnProperty(prefix = "app.storage", name = "mode", havingValue = "mock", matchIfMissing = true)
public class MockUploadController {

    @PutMapping("/mock-upload/**")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    void upload(
        HttpServletRequest request,
        @RequestHeader(value = "Content-Type", required = false) String contentType,
        @RequestBody(required = false) byte[] body
    ) {
        int byteLength = body == null ? 0 : body.length;
        log.info(
            "Accepted mock upload: path={}, contentType={}, bytes={}",
            request.getRequestURI(),
            contentType,
            byteLength
        );
    }
}
