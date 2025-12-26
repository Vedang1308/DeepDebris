package com.deepdebris.backend.service;

import com.deepdebris.backend.model.TleData;
import com.influxdb.client.InfluxDBClient;
import com.influxdb.client.WriteApiBlocking;
import com.influxdb.client.domain.WritePrecision;
import com.influxdb.client.write.Point;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
@RequiredArgsConstructor
@Slf4j
public class TlePersister {

    private final InfluxDBClient influxDBClient;

    @KafkaListener(topics = "tle-updates", groupId = "deepdebris-persister")
    public void consumeTle(TleData tleData) {
        log.info("Persisting TLE for {}", tleData.getNoradId());

        try {
            WriteApiBlocking writeApi = influxDBClient.getWriteApiBlocking();

            // InfluxDB Point
            // Measurement: orbital_elements
            // Tags: norad_id, name
            // Fields: raw_line1, raw_line2 (storing as strings is okay for logs, but Influx
            // is numeric optimized)
            // Storing raw TLE strings in Influx is not ideal for querying, but useful for
            // history re-play.

            Point point = Point.measurement("rv_vectors") // renaming to generic
                    .addTag("norad_id", tleData.getNoradId())
                    .addTag("name", tleData.getName())
                    .addField("line1", tleData.getLine1()) // Influx supports string fields
                    .addField("line2", tleData.getLine2())
                    .time(Instant.now(), WritePrecision.MS);

            writeApi.writePoint(point);

        } catch (Exception e) {
            log.error("Failed to write to InfluxDB", e);
        }
    }
}
