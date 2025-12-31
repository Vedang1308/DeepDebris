package com.deepdebris.backend.service;

import com.deepdebris.backend.model.TleData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class SpaceTrackService {

    private final TleProducer tleProducer;

    @org.springframework.beans.factory.annotation.Value("${spacetrack.user:}")
    private String username;

    @org.springframework.beans.factory.annotation.Value("${spacetrack.password:}")
    private String password;

    // Simulate fetching TLEs every 10 seconds
    @Scheduled(fixedRate = 10000)
    public void fetchTleData() {
        if (username != null && !username.isEmpty() && password != null && !password.isEmpty()) {
            fetchRealData();
        } else {
            log.error(
                    "CRITICAL: No Space-Track credentials provided. Cannot fetch REAL data. System will not function.");
            throw new RuntimeException("Space-Track credentials missing. Strict Real Data Mode active.");
        }
    }

    private void fetchRealData() {
        log.info("Attempting to fetch REAL data from Space-Track.org for user: {}", username);

        try {
            org.springframework.web.client.RestTemplate restTemplate = new org.springframework.web.client.RestTemplate();
            String loginUrl = "https://www.space-track.org/ajaxauth/login";

            // Space-Track API expects a POST with identity, password, and the query itself
            // in the body/params
            // to auto-login and fetch in one go.

            // Query: Get latest TLEs for top 100 objects (or specific ones to save
            // bandwidth)
            // For demo: ISS (25544) and Hubble (20580)
            String query = "https://www.space-track.org/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/25544,20580/orderby/ORDINAL desc/format/json";

            org.springframework.util.MultiValueMap<String, String> map = new org.springframework.util.LinkedMultiValueMap<>();
            map.add("identity", username);
            map.add("password", password);
            map.add("query", query);

            TleData[] response = restTemplate.postForObject(loginUrl, map, TleData[].class);

            if (response != null) {
                log.info("Successfully fetched {} TLEs from Space-Track.", response.length);
                for (TleData tle : response) {
                    tleProducer.sendTleUpdate(tle);
                }
            }
        } catch (Exception e) {
            log.error("Error fetching from Space-Track: {}", e.getMessage());
            throw new RuntimeException("Failed to fetch REAL data from Space-Track.", e);
        }
    }

}
