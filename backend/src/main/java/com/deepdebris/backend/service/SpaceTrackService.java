package com.deepdebris.backend.service;

import com.deepdebris.backend.model.TleData;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;

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
            log.info("Fetching TLE data (mocked) - No credentials provided.");
            List<TleData> mockData = generateMockData();
            mockData.forEach(tleProducer::sendTleUpdate);
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
            // Fallback
            List<TleData> mockData = generateMockData();
            mockData.forEach(tleProducer::sendTleUpdate);
        }
    }

    private List<TleData> generateMockData() {
        // Return some dummy TLEs (ISS, Hubble)
        return List.of(
                new TleData("25544", "ISS (ZARYA)",
                        "1 25544U 98067A   23356.54321689  .00016717  00000+0  30283-3 0  9997",
                        "2 25544  51.6416  21.9684 0005432  35.2163  86.1264 15.49507156430342",
                        Instant.now().toString()),
                new TleData("20580", "HUBBLE SPACE TELESCOPE",
                        "1 20580U 90037B   23355.12345678  .00001234  00000+0  12345-4 0  9993",
                        "2 20580  28.4699 123.4567 0001234  12.3456 123.4567 15.09345678 12345",
                        Instant.now().toString()));
    }
}
