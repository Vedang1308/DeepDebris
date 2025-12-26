package com.deepdebris.backend.config;

import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.InfluxDBClientFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class InfluxDbConfig {

    @Value("${influxdb.url:http://localhost:8086}")
    private String url;

    @Value("${influxdb.token:adminpassword}") // In prod, this should be a token
    private String token; // For InfluxDB v2, this is effectively used as password if auth enabled or
                          // token

    @Value("${influxdb.org:deepdebris}")
    private String org;

    @Value("${influxdb.bucket:tle_history}")
    private String bucket;

    @Bean
    public InfluxDBClient influxDBClient() {
        // For simple username/password auth in compat mode or token
        // In this docker setup, we are using init-mode=setup with admin/adminpassword
        // The Java client usually expects a token for v2.
        // We will assume a token is generated or we use the "token" generic auth.
        // For MVP docker setup, we might need to grab the token from CLI or hardcode a
        // known token if possible.
        // Alternatively, use v1 compat API if needed, but v2 is standard.
        // For simplicity in this demo, let's assume we pass the raw pointer.

        return InfluxDBClientFactory.create(url, token.toCharArray(), org, bucket);
    }
}
