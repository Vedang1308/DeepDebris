package com.deepdebris.backend.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@com.fasterxml.jackson.annotation.JsonIgnoreProperties(ignoreUnknown = true)
public class TleData {
    @com.fasterxml.jackson.annotation.JsonAlias("NORAD_CAT_ID")
    private String noradId;

    @com.fasterxml.jackson.annotation.JsonAlias("OBJECT_NAME")
    private String name;

    @com.fasterxml.jackson.annotation.JsonAlias("TLE_LINE1")
    private String line1;

    @com.fasterxml.jackson.annotation.JsonAlias("TLE_LINE2")
    private String line2;

    @com.fasterxml.jackson.annotation.JsonAlias("EPOCH")
    private String epoch;
}
