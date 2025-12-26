package com.deepdebris.backend.service;

import com.deepdebris.backend.model.TleData;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Service
@RequiredArgsConstructor
@Slf4j
public class TleProducer {

    private final KafkaTemplate<String, TleData> kafkaTemplate;

    public void sendTleUpdate(TleData tleData) {
        log.info("Sending TLE update for {}", tleData.getNoradId());
        kafkaTemplate.send("tle-updates", tleData.getNoradId(), tleData);
    }
}
