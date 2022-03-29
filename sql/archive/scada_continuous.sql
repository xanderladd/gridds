CREATE TABLE scada_continuous (
    datetime timestamp without time zone NOT NULL,
    element_id varchar(32),
    device_type varchar(32),
    kwh double  precision,
    co_op varchar(32)
);

-- CREATE INDEX idx_scada_continuous ON scada_continuous(element_id);

SELECT create_hypertable('scada_continuous', 'datetime');
SELECT set_chunk_time_interval('scada_continuous', INTERVAL '15 Days');
SELECT add_dimension('scada_continuous', 'element_id', number_partitions => 82);
