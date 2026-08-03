package com.springboot.hair_customized_ai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class HairCustomizedAiApplication {

    public static void main(String[] args) {
        SpringApplication.run(HairCustomizedAiApplication.class, args);
    }

}
